from __future__ import annotations
from typing import Union
import copy
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from segment_anything import SamAutomaticMaskGenerator
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from segment_anything.modeling.mask_decoder import MaskDecoder
from segment_anything.modeling.prompt_encoder import PromptEncoder
from segment_anything.utils.amg import build_all_layer_point_grids
from segment_anything.utils.transforms import ResizeLongestSide
from torchvision.transforms.functional import resize, to_pil_image

from efficientvit.models.efficientvit.backbone import EfficientViTBackbone, EfficientViTLargeBackbone
from efficientvit.models.nn import (
    ConvLayer,
    DAGBlock,
    FusedMBConv,
    IdentityLayer,
    MBConv,
    OpSequential,
    ResBlock,
    ResidualBlock,
    UpSampleLayer,
    build_norm,
)
from efficientvit.models.utils import build_kwargs_from_config, get_device

__all__ = [
    "SamPad",
    "SamResize",
    "SamNeck",
    "EfficientViTSamImageEncoder",
    "EfficientViTSam",
    "EfficientViTSamPredictor",
    "EfficientViTSamAutomaticMaskGenerator",
    "efficientvit_sam_l0",
    "efficientvit_sam_l1",
    "efficientvit_sam_l2",
    "efficientvit_sam_xl0",
    "efficientvit_sam_xl1",
]


class SamPad:
    def __init__(self, size: int, fill: float = 0, pad_mode="corner") -> None:
        self.size = size
        self.fill = fill
        self.pad_mode = pad_mode

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        h, w = image.shape[-2:]
        th, tw = self.size, self.size
        assert th >= h and tw >= w
        if self.pad_mode == "corner":
            image = F.pad(image, (0, tw - w, 0, th - h), value=self.fill)
        else:
            raise NotImplementedError
        return image

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size},mode={self.pad_mode},fill={self.fill})"


class SamResize:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        h, w, _ = image.shape
        long_side = max(h, w)
        if long_side != self.size:
            return self.apply_image(image)
        else:
            return image

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.size)
        return np.array(resize(to_pil_image(image), target_size))

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size})"


class SamNeck(DAGBlock):
    def __init__(
        self,
        fid_list: list[str],
        in_channel_list: list[int],
        head_width: int,
        head_depth: int,
        expand_ratio: float,
        middle_op: str,
        out_dim: int = 256,
        norm="bn2d",
        act_func="gelu",
    ):
        inputs = {}
        for fid, in_channel in zip(fid_list, in_channel_list):
            inputs[fid] = OpSequential(
                [
                    ConvLayer(in_channel, head_width, 1, norm=norm, act_func=None),
                    UpSampleLayer(size=(64, 64)),
                ]
            )

        middle = []
        for _ in range(head_depth):
            if middle_op == "mb":
                block = MBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, act_func, None),
                )
            elif middle_op == "fmb":
                block = FusedMBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, None),
                )
            elif middle_op == "res":
                block = ResBlock(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, None),
                )
            else:
                raise NotImplementedError
            middle.append(ResidualBlock(block, IdentityLayer()))
        middle = OpSequential(middle)

        outputs = {
            "sam_encoder": OpSequential(
                [
                    ConvLayer(
                        head_width,
                        out_dim,
                        1,
                        use_bias=True,
                        norm=None,
                        act_func=None,
                    ),
                ]
            )
        }

        super(SamNeck, self).__init__(inputs, "add", None, middle=middle, outputs=outputs)


class EfficientViTSamImageEncoder(nn.Module):
    def __init__(self, backbone: Union[EfficientViTBackbone, EfficientViTLargeBackbone], neck: SamNeck):
        super().__init__()
        self.backbone = backbone
        self.neck = neck

        self.norm = build_norm("ln2d", 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feed_dict = self.backbone(x)
        feed_dict = self.neck(feed_dict)

        output = feed_dict["sam_encoder"]
        output = self.norm(output)
        return output


class EfficientViTSam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: EfficientViTSamImageEncoder,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        image_size: tuple[int, int] = (1024, 512),
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

        self.image_size = image_size

        self.transform = transforms.Compose(
            [
                SamResize(self.image_size[1]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[123.675 / 255, 116.28 / 255, 103.53 / 255],
                    std=[58.395 / 255, 57.12 / 255, 57.375 / 255],
                ),
                SamPad(self.image_size[1]),
            ]
        )

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: tuple[int, ...],
        original_size: tuple[int, ...],
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (self.image_size[0], self.image_size[0]),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def forward(
        self,
        batched_input: list[dict[str, Any]],
        multimask_output: bool,
    ):
        input_images = torch.stack([x["image"] for x in batched_input], dim=0)

        image_embeddings = self.image_encoder(input_images)

        outputs = []
        iou_outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            outputs.append(low_res_masks)
            iou_outputs.append(iou_predictions)

        outputs = torch.stack([out for out in outputs], dim=0)
        iou_outputs = torch.stack(iou_outputs, dim=0)

        return outputs, iou_outputs


class EfficientViTSamPredictor:
    """
    GPU-first predictor for EfficientViT-SAM.

    New in this version:
      • set_image_cuda(img_hwc_u8_cuda, image_format="RGB"): preprocess & encode fully on GPU
      • predict_points_cuda(point_coords_hw, point_labels): convenience wrapper for point prompts on GPU
      • GPU transform (resize → normalize → pad) implemented with torch.nn.functional

    Backwards-compatible:
      • set_image(np.ndarray) and predict(...) still work (CPU path)
    """
    def __init__(self, sam_model: EfficientViTSam) -> None:
        self.model = sam_model
        self.reset_image()

    @property
    def device(self):
        return get_device(self.model)

    # --------------------------
    # state / bookkeeping
    # --------------------------
    def reset_image(self) -> None:
        self.is_image_set = False
        self.features = None            # (1, 256, 64, 64)
        self.original_size = None       # (H_orig, W_orig)
        self.input_size = None          # (H_rs1024, W_rs1024) for prompt scaling

    # --------------------------
    # coordinate scaling helpers
    # --------------------------
    def apply_coords(self, coords: np.ndarray, im_size=None) -> np.ndarray:
        """CPU version used by legacy predict(). Scales XY from original_size -> input_size (1024 long side)."""
        old_h, old_w = self.original_size
        new_h, new_w = self.input_size
        coords = copy.deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, im_size=None) -> np.ndarray:
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2))
        return boxes.reshape(-1, 4)

    def apply_coords_torch(self, coords: torch.Tensor, im_size=None) -> torch.Tensor:
        """GPU version for prompt scaling (original_size -> input_size@1024)."""
        assert self.original_size is not None and self.input_size is not None
        old_h, old_w = self.original_size
        new_h, new_w = self.input_size
        coords_copy = coords.to(dtype=torch.float32)
        coords_copy[..., 0] = coords_copy[..., 0] * (float(new_w) / float(old_w))
        coords_copy[..., 1] = coords_copy[..., 1] * (float(new_h) / float(old_h))
        return coords_copy

    def apply_boxes_torch(self, boxes: torch.Tensor) -> torch.Tensor:
        boxes2x2 = self.apply_coords_torch(boxes.reshape(-1, 2, 2))
        return boxes2x2.reshape(-1, 4)

    # --------------------------
    # CPU path (unchanged APIs)
    # --------------------------
    @torch.inference_mode()
    def set_image(self, image: np.ndarray, image_format: str = "RGB") -> None:
        assert image_format in ["RGB", "BGR"], f"image_format must be in ['RGB','BGR'], got {image_format}"
        if image_format != self.model.image_format:
            image = image[..., ::-1]  # BGR->RGB

        self.reset_image()
        self.original_size = image.shape[:2]

        # prompt-scaling reference uses LONGSIDE=1024
        self.input_size = ResizeLongestSide.get_preprocess_shape(
            *self.original_size, long_side_length=self.model.image_size[0]
        )

        # CPU transform to SxS (S=self.model.image_size[1])
        torch_data = self.model.transform(image).unsqueeze(0).to(self.device)
        self.features = self.model.image_encoder(torch_data)
        self.is_image_set = True

    @torch.inference_mode()
    def set_image_batch(self, image: torch.Tensor) -> None:
        """
        Expect (B,C,H,W) already preprocessed to SxS (see EfficientViTSam.transform equivalent).
        """
        self.reset_image()
        original_height, original_width = image.shape[-2], image.shape[-1]
        self.original_size = (int(original_height), int(original_width))
        self.input_size = ResizeLongestSide.get_preprocess_shape(
            *self.original_size, long_side_length=self.model.image_size[0]
        )
        self.features = self.model.image_encoder(image.to(self.device))
        self.is_image_set = True

    # --------------------------
    # NEW: GPU transform & set_image_cuda
    # --------------------------
    @staticmethod
    def _normalize_chw(img_chw: torch.Tensor) -> torch.Tensor:
        """
        img_chw: (C,H,W) float32 in [0,1], on CUDA.
        Apply SAM mean/std (RGB).
        """
        mean = torch.tensor([123.675/255.0, 116.28/255.0, 103.53/255.0],
                            device=img_chw.device, dtype=img_chw.dtype)[:, None, None]
        std  = torch.tensor([58.395/255.0, 57.12/255.0, 57.375/255.0],
                            device=img_chw.device, dtype=img_chw.dtype)[:, None, None]
        return (img_chw - mean) / std

    @torch.inference_mode()
    def set_image_cuda(self, image_hwc_u8: torch.Tensor, image_format: str = "RGB") -> None:
        """
        Fully-GPU path. Input:
          image_hwc_u8: CUDA tensor (H,W,3) uint8 mapped from a cudaImage (zero-copy).
        """
        if not isinstance(image_hwc_u8, torch.Tensor):
            raise TypeError("set_image_cuda expects a CUDA torch.Tensor")
        if image_hwc_u8.device.type != "cuda" or image_hwc_u8.dtype != torch.uint8 or image_hwc_u8.dim() != 3:
            raise ValueError("image must be CUDA uint8 with shape (H,W,3)")

        H, W, C = image_hwc_u8.shape
        assert C == 3, "expected 3 channels"
        self.reset_image()
        self.original_size = (int(H), int(W))

        # prompt-scaling grid uses long_side=1024 (model.image_size[0])
        self.input_size = ResizeLongestSide.get_preprocess_shape(
            *self.original_size, long_side_length=self.model.image_size[0]
        )

        # convert to float [0,1], possibly swap BGR->RGB
        img = image_hwc_u8.to(torch.float32) / 255.0  # (H,W,3) float32
        if image_format != self.model.image_format:
            # BGR -> RGB
            img = img[..., [2, 1, 0]]

        # resize long side to S = image_size[1] (encoder input size), keep aspect
        S = int(self.model.image_size[1])  # e.g., 512
        long_side = max(H, W)
        if long_side == 0:
            raise ValueError("invalid image with zero side")
        if long_side != S:
            scale = float(S) / float(long_side)
            new_h = int(round(H * scale))
            new_w = int(round(W * scale))
        else:
            new_h, new_w = H, W

        # to CHW, add batch
        img_chw = img.permute(2, 0, 1).contiguous()  # (3,H,W)

        # bilinear resize on GPU to (new_h,new_w)
        if (new_h, new_w) != (H, W):
            img_chw = F.interpolate(img_chw[None, ...], size=(new_h, new_w), mode="bilinear", align_corners=False)[0]

        # normalize
        img_chw = self._normalize_chw(img_chw)  # (3,new_h,new_w)

        # pad bottom/right to SxS (corner mode)
        pad_h = S - new_h
        pad_w = S - new_w
        if pad_h < 0 or pad_w < 0:
            raise RuntimeError("computed negative padding; check resizing logic")
        if pad_h or pad_w:
            img_chw = F.pad(img_chw, (0, pad_w, 0, pad_h))  # pad=(left,right,top,bottom) on CHW → (0,r,0,b)

        # encode
        torch_data = img_chw[None, ...].to(self.device)      # (1,3,S,S)
        self.features = self.model.image_encoder(torch_data) # (1,256,64,64)
        self.is_image_set = True

    # --------------------------
    # Prediction APIs
    # --------------------------
    def _post_masks(self, low_res_masks: torch.Tensor) -> torch.Tensor:
        """Helper just to centralize mask postprocess + thresholding."""
        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)
        return masks > self.model.mask_threshold

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """CPU-friendly wrapper (unchanged signature)."""
        if not self.is_image_set:
            raise RuntimeError("Call set_image(...) or set_image_cuda(...) first.")
        device = self.device
        coords_t, labels_t, box_t, mask_in_t = None, None, None, None
        if point_coords is not None:
            assert point_labels is not None
            coords = self.apply_coords(point_coords)
            coords_t = torch.as_tensor(coords, dtype=torch.float32, device=device)[None, ...]  # (1,N,2)
            labels_t = torch.as_tensor(point_labels, dtype=torch.int64, device=device)[None, ...]  # (1,N)
        if box is not None:
            box = self.apply_boxes(box)
            box_t = torch.as_tensor(box, dtype=torch.float32, device=device)[None, ...]
        if mask_input is not None:
            mask_in_t = torch.as_tensor(mask_input, dtype=torch.float32, device=device)[None, ...]  # (1,1,256,256)

        masks, iou_pred, lowres = self.predict_torch(
            coords_t, labels_t, box_t, mask_in_t, multimask_output, return_logits=return_logits
        )
        # Return numpy for legacy behavior
        return (
            masks[0].detach().cpu().numpy(),
            iou_pred[0].detach().cpu().numpy(),
            lowres[0].detach().cpu().numpy(),
        )

    @torch.inference_mode()
    def predict_torch(
        self,
        point_coords: Optional[torch.Tensor] = None,   # (B,N,2) already scaled to 1024 grid
        point_labels: Optional[torch.Tensor] = None,   # (B,N)
        boxes: Optional[torch.Tensor] = None,          # (B,4) on 1024 grid
        mask_input: Optional[torch.Tensor] = None,     # (B,1,256,256)
        multimask_output: bool = True,
        return_logits: bool = False,
        image_index: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.is_image_set:
            raise RuntimeError("Call set_image(...) or set_image_cuda(...) first.")

        # prompts → embeddings
        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points, boxes=boxes, masks=mask_input
        )

        # choose embedding for image_index (supports set_image_batch)
        if image_index is not None:
            image_embeddings = self.features[image_index].unsqueeze(0)
        else:
            image_embeddings = self.features

        # decode masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # upsample to original resolution
        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)
        if not return_logits:
            masks = masks > self.model.mask_threshold
        return masks, iou_predictions, low_res_masks

    # --------------------------
    # NEW: Convenience (GPU points)
    # --------------------------
    @torch.inference_mode()
    def predict_points_cuda(
        self,
        point_coords_hw: torch.Tensor,     # (N,2) in ORIGINAL image pixels (H,W)
        point_labels: torch.Tensor,        # (N,) 0/1
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pass prompts as CUDA tensors in ORIGINAL pixel coordinates.
        Handles scaling to the 1024 grid internally on GPU.
        Returns masks (1,C,H,W) at ORIGINAL resolution on CUDA.
        """
        if point_coords_hw.numel() == 0:
            return self.predict_torch(None, None, None, None, multimask_output, return_logits)

        # scale to the 1024 reference grid on GPU
        coords = self.apply_coords_torch(point_coords_hw)         # (N,2)
        coords = coords[None, ...].to(self.device)               # (1,N,2)
        labels = point_labels.to(self.device).to(torch.int64)[None, ...]  # (1,N)

        return self.predict_torch(coords, labels, None, None, multimask_output, return_logits)

    @torch.inference_mode()
    def predict_boxes_cuda(
        self,
        boxes_xyxy: torch.Tensor,          # (N,4) in ORIGINAL image pixels (xyxy), CUDA/float32
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        GPU path for box prompts.
        Returns:
          masks_best : (N,H,W) bool (or logits if return_logits=True)
          iou_best   : (N,)     float32
        """
        if not isinstance(boxes_xyxy, torch.Tensor):
            raise TypeError("boxes_xyxy must be a torch.Tensor")
        if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[1] != 4:
            raise ValueError("boxes_xyxy must have shape (N,4)")
        if boxes_xyxy.device.type != "cuda":
            boxes_xyxy = boxes_xyxy.to(self.device)
        boxes_xyxy = boxes_xyxy.to(torch.float32)

        N = boxes_xyxy.shape[0]
        if N == 0:
            H, W = self.original_size
            empty = torch.empty((0, int(H), int(W)), device=self.device, dtype=torch.bool)
            return empty, torch.empty((0,), device=self.device, dtype=torch.float32)

        # scale to 1024 reference grid on GPU
        boxes_scaled = self.apply_boxes_torch(boxes_xyxy)  # (N,4) float32 CUDA

        masks_best = []
        iou_best   = []

        # process one box at a time (predict_torch expects (B,4) and B==1)
        for i in range(N):
            box_i = boxes_scaled[i].unsqueeze(0)  # (1,4)
            masks, iou_pred, lowres = self.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=box_i,
                mask_input=None,
                multimask_output=multimask_output,
                return_logits=return_logits,
            )  # masks: (1,K,H,W), iou_pred: (1,K)

            # pick best proposal for this box
            idx = torch.argmax(iou_pred[0]).item()
            best_m = masks[0, idx]  # (H,W)
            best_s = iou_pred[0, idx]

            masks_best.append(best_m)
            iou_best.append(best_s)

        masks_best = torch.stack(masks_best, dim=0)  # (N,H,W)
        iou_best   = torch.stack(iou_best, dim=0)    # (N,)
        return masks_best, iou_best


class EfficientViTSamAutomaticMaskGenerator(SamAutomaticMaskGenerator):
    def __init__(
        self,
        model: EfficientViTSam,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[list[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
    ) -> None:
        assert (points_per_side is None) != (
            point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."

        self.predictor = EfficientViTSamPredictor(model)
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode


def build_efficientvit_sam(image_encoder: EfficientViTSamImageEncoder, image_size: int) -> EfficientViTSam:
    return EfficientViTSam(
        image_encoder=image_encoder,
        prompt_encoder=PromptEncoder(
            embed_dim=256,
            image_embedding_size=(64, 64),
            input_image_size=(1024, 1024),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        image_size=(1024, image_size),
    )


def efficientvit_sam_l0(image_size: int = 512, **kwargs) -> EfficientViTSam:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_l0

    backbone = efficientvit_backbone_l0(**kwargs)

    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[512, 256, 128],
        head_width=256,
        head_depth=4,
        expand_ratio=1,
        middle_op="fmb",
    )

    image_encoder = EfficientViTSamImageEncoder(backbone, neck)
    return build_efficientvit_sam(image_encoder, image_size)


def efficientvit_sam_l1(image_size: int = 512, **kwargs) -> EfficientViTSam:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_l1

    backbone = efficientvit_backbone_l1(**kwargs)

    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[512, 256, 128],
        head_width=256,
        head_depth=8,
        expand_ratio=1,
        middle_op="fmb",
    )

    image_encoder = EfficientViTSamImageEncoder(backbone, neck)
    return build_efficientvit_sam(image_encoder, image_size)


def efficientvit_sam_l2(image_size: int = 512, **kwargs) -> EfficientViTSam:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_l2

    backbone = efficientvit_backbone_l2(**kwargs)

    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[512, 256, 128],
        head_width=256,
        head_depth=12,
        expand_ratio=1,
        middle_op="fmb",
    )

    image_encoder = EfficientViTSamImageEncoder(backbone, neck)
    return build_efficientvit_sam(image_encoder, image_size)


def efficientvit_sam_xl0(image_size: int = 1024, **kwargs) -> EfficientViTSam:
    from efficientvit.models.efficientvit.backbone import EfficientViTLargeBackbone

    backbone = EfficientViTLargeBackbone(
        width_list=[32, 64, 128, 256, 512, 1024],
        depth_list=[0, 1, 1, 2, 3, 3],
        block_list=["res", "fmb", "fmb", "fmb", "att@3", "att@3"],
        expand_list=[1, 4, 4, 4, 4, 6],
        fewer_norm_list=[False, False, False, False, True, True],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )

    neck = SamNeck(
        fid_list=["stage5", "stage4", "stage3"],
        in_channel_list=[1024, 512, 256],
        head_width=256,
        head_depth=6,
        expand_ratio=4,
        middle_op="fmb",
    )

    image_encoder = EfficientViTSamImageEncoder(backbone, neck)
    return build_efficientvit_sam(image_encoder, image_size)


def efficientvit_sam_xl1(image_size: int = 1024, **kwargs) -> EfficientViTSam:
    from efficientvit.models.efficientvit.backbone import EfficientViTLargeBackbone

    backbone = EfficientViTLargeBackbone(
        width_list=[32, 64, 128, 256, 512, 1024],
        depth_list=[1, 2, 2, 4, 6, 6],
        block_list=["res", "fmb", "fmb", "fmb", "att@3", "att@3"],
        expand_list=[1, 4, 4, 4, 4, 6],
        fewer_norm_list=[False, False, False, False, True, True],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )

    neck = SamNeck(
        fid_list=["stage5", "stage4", "stage3"],
        in_channel_list=[1024, 512, 256],
        head_width=256,
        head_depth=12,
        expand_ratio=4,
        middle_op="fmb",
    )

    image_encoder = EfficientViTSamImageEncoder(backbone, neck)
    return build_efficientvit_sam(image_encoder, image_size)