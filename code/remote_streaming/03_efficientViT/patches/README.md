[ [English](#english) | [한국어](#한국어) ]

# EfficientViT — Patches (To‑Do for your conda venv)
**Up one level:** [`../README.md`](../README.md)
**Prerequisite:** install jetson-inference on conda venv first → [`../../../jetson_inference_patch/README.md`](../../../jetson_inference_patch/README.md)

---

## English

Minimal checklist to run **EfficientViT** on Jetson (Python 3.8) with CUDA‑friendly SAM.

### 1) Install pure‑Python deps first
```bash
# EfficientViT deps
pip install einops timm tqdm torchprofile torchmetrics scipy torch-fidelity torchdiffeq diffusers omegaconf ipdb "wandb[media]" matplotlib huggingface-hub transformers pycocotools lvis scikit-image gradio

# TinyNeuralNetwork (no-deps)
pip install "PyYAML>=5.3.1" "ruamel.yaml>=0.16.12" "igraph>=0.9" "flatbuffers>=1.12"
pip install --no-deps git+https://github.com/alibaba/TinyNeuralNetwork.git

# Segment Anything (no-deps)
pip install --no-deps git+https://github.com/facebookresearch/segment-anything.git
```

### 2) ONNX toolchain
```bash
pip install onnx onnxruntime
```

### 3) Editable install of EfficientViT (no deps, no build isolation)
```bash
cd ~
git clone https://github.com/mit-han-lab/efficientvit.git
cd efficientvit
pip install --no-deps --no-build-isolation -e .
```

### 4) Apply Jetson/Py3.8 + SAM patches
```bash
cd ~/ir-guide/code/remote_streaming/03_efficientViT/patches
python pep604_patch.py --root ~/efficientvit/efficientvit
python add_future_annotations.py --root ~/efficientvit/efficientvit
python copy_patch_efficientvit.py --root ~/efficientvit/efficientvit
```

_Tip:_ If your shell is zsh, quotes around `wandb[media]` avoid globbing.

---

## 한국어

**EfficientViT** 를 Jetson(Python 3.8)에서 **CUDA 호환 SAM**과 함께 실행하기 위한 최소 체크리스트입니다.

### 1) 순수 파이썬 의존성 먼저 설치
```bash
# EfficientViT 필수
pip install einops timm tqdm torchprofile torchmetrics scipy torch-fidelity torchdiffeq diffusers omegaconf ipdb "wandb[media]" matplotlib huggingface-hub transformers pycocotools lvis scikit-image gradio

# TinyNeuralNetwork (의존성 없이)
pip install "PyYAML>=5.3.1" "ruamel.yaml>=0.16.12" "igraph>=0.9" "flatbuffers>=1.12"
pip install --no-deps git+https://github.com/alibaba/TinyNeuralNetwork.git

# Segment Anything (의존성 없이)
pip install --no-deps git+https://github.com/facebookresearch/segment-anything.git
```

### 2) ONNX 도구체인
```bash
pip install onnx onnxruntime
```

### 3) EfficientViT 편집 모드 설치(no-deps, build isolation 해제)
```bash
cd ~
git clone https://github.com/mit-han-lab/efficientvit.git
cd efficientvit
pip install --no-deps --no-build-isolation -e .
```

### 4) Jetson/Py3.8 + SAM 패치 적용
```bash
cd ~/ir-guide/code/remote_streaming/03_efficientViT/patches
python pep604_patch.py --root ~/efficientvit/efficientvit
python add_future_annotations.py --root ~/efficientvit/efficientvit
python copy_patch_efficientvit.py --root ~/efficientvit/efficientvit
```

_메모:_ zsh를 사용한다면 `wandb[media]`에 따옴표를 붙여 글로빙을 방지하세요.