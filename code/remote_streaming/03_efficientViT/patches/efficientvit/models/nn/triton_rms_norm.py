# efficientvit/models/nn/triton_rms_norm.py
from __future__ import annotations
import torch

__all__ = ["TritonRMSNorm2dFunc"]

# Try to use Triton if available; otherwise fall back to a pure-PyTorch autograd.
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False


if _HAS_TRITON:
    @triton.jit
    def _rms_norm_2d_fwd_fused(
        X,  # pointer to the input
        Y,  # pointer to the output
        W,  # pointer to the weights
        B,  # pointer to the biases
        Rrms,  # pointer to the 1/rms
        M,
        C,
        N,
        num_blocks,  # number of columns in X
        eps,  # epsilon to avoid division by zero
        BLOCK_SIZE: tl.constexpr,
    ):
        # Map the program id to the row of X and Y it should compute.
        m_n = tl.program_id(0)
        m, n = m_n // num_blocks, m_n % num_blocks

        Y += m * C * N
        X += m * C * N

        cols = n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        x_sum_square = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, C):
            x = tl.load(X + off * N + cols, mask=mask, other=0.0).to(tl.float32)
            x_sum_square += x * x
        mean_square = x_sum_square / C
        rrms = 1 / tl.sqrt(mean_square + eps)
        tl.store(Rrms + m * N + cols, rrms, mask=mask)

        for off in range(0, C):
            pos = off * N + cols
            w = tl.load(W + off)
            b = tl.load(B + off)
            x = tl.load(X + pos, mask=mask, other=0.0).to(tl.float32)
            x_hat = x * rrms
            y = x_hat * w + b
            tl.store(Y + pos, y, mask=mask)

    @triton.jit
    def _rms_norm_2d_bwd_dx_fused(
        DX,  # pointer to the input gradient
        DY,  # pointer to the output gradient
        DW,  # pointer to the partial sum of weights gradient
        DB,  # pointer to the partial sum of biases gradient
        X,   # pointer to the input
        W,   # pointer to the weights
        B,   # pointer to the biases
        Rrms,  # pointer to the 1/rms
        M,
        C,
        N,  # number of columns in X
        num_blocks,
        eps,  # epsilon to avoid division by zero
        GROUP_SIZE_M: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_SIZE_C: tl.constexpr,
    ):
        m_n = tl.program_id(0)
        m, n = m_n // num_blocks, m_n % num_blocks
        X += m * C * N
        DY += m * C * N
        DX += m * C * N
        Rrms += m * N

        cols = n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        DW = DW + m_n * C
        DB = DB + m_n * C
        rrms = tl.load(Rrms + cols, mask=mask, other=1)

        c1 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, C):
            pos = off * N + cols
            x = tl.load(X + pos, mask=mask, other=0).to(tl.float32)
            dy = tl.load(DY + pos, mask=mask, other=0).to(tl.float32)
            w = tl.load(W + off).to(tl.float32)
            xhat = x * rrms
            wdy = w * dy
            xhat = tl.where(mask, xhat, 0.0)
            wdy  = tl.where(mask, wdy, 0.0)
            c1 += xhat * wdy
            tl.store(DW + off, tl.sum((dy * xhat).to(w.dtype), axis=0))
            tl.store(DB + off, tl.sum(dy.to(w.dtype), axis=0))

        c1 /= C
        for off in range(0, C):
            pos = off * N + cols
            x = tl.load(X + pos, mask=mask, other=0).to(tl.float32)
            dy = tl.load(DY + pos, mask=mask, other=0).to(tl.float32)
            w = tl.load(W + off).to(tl.float32)
            xhat = x * rrms
            wdy  = w * dy
            dx = (wdy - (xhat * c1)) * rrms
            tl.store(DX + pos, dx, mask=mask)

    class TritonRMSNorm2dFunc(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, bias, eps):
            y = torch.empty_like(x)
            x_arg = x.reshape(x.shape[0], x.shape[1], -1)
            M, C, N = x_arg.shape
            rrms = torch.empty((M, N), dtype=torch.float32, device=x.device)

            BLOCK_SIZE = 256
            num_blocks = triton.cdiv(N, BLOCK_SIZE)
            num_warps = 8

            _rms_norm_2d_fwd_fused[(M * num_blocks,)](
                x_arg, y, weight, bias, rrms,
                M, C, N, num_blocks, eps,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
                num_ctas=1,
            )
            ctx.save_for_backward(x, weight, bias, rrms)
            ctx.BLOCK_SIZE = BLOCK_SIZE
            ctx.num_blocks = num_blocks
            ctx.num_warps = num_warps
            ctx.eps = eps
            return y

        @staticmethod
        def backward(ctx, dy):
            x, w, b, rrms = ctx.saved_tensors
            num_blocks = ctx.num_blocks

            x_arg = x.reshape(x.shape[0], x.shape[1], -1)
            M, C, N = x_arg.shape
            GROUP_SIZE_M = M * num_blocks

            _dw = torch.empty((GROUP_SIZE_M, C), dtype=x.dtype, device=w.device)
            _db = torch.empty((GROUP_SIZE_M, C), dtype=x.dtype, device=w.device)
            dw  = torch.empty((C,), dtype=w.dtype, device=w.device)
            db  = torch.empty((C,), dtype=w.dtype, device=w.device)
            dx  = torch.empty_like(dy)

            _rms_norm_2d_bwd_dx_fused[(M * num_blocks,)](
                dx, dy, _dw, _db, x, w, b, rrms,
                M, C, N, num_blocks, ctx.eps,
                BLOCK_SIZE=ctx.BLOCK_SIZE,
                GROUP_SIZE_M=GROUP_SIZE_M,
                BLOCK_SIZE_C=triton.next_power_of_2(C),
                num_warps=ctx.num_warps,
            )
            dw = _dw.sum(dim=0)
            db = _db.sum(dim=0)
            return dx, dw, db, None

else:
    # --------- Pure-PyTorch fallback (works on Jetson without Triton) ----------
    class TritonRMSNorm2dFunc(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, bias, eps):
            """
            RMSNorm over channel dimension for 4D input (N, C, H, W):
              r = 1 / sqrt(mean(x^2, dim=1, keepdim=True) + eps)
              y = (x * r) * weight + bias
            """
            # compute r in float32 for stability, then cast back
            x_f = x.float()
            rms = torch.sqrt(x_f.pow(2).mean(dim=1, keepdim=True) + eps)
            r = (1.0 / rms).to(x.dtype)

            # affine params
            if weight is None:
                w = torch.ones((x.shape[1],), dtype=x.dtype, device=x.device)
            else:
                w = weight.to(dtype=x.dtype, device=x.device)
            if bias is None:
                b = torch.zeros((x.shape[1],), dtype=x.dtype, device=x.device)
            else:
                b = bias.to(dtype=x.dtype, device=x.device)

            y = (x * r) * w.view(1, -1, 1, 1) + b.view(1, -1, 1, 1)

            # save for backward
            ctx.save_for_backward(x, r, w)
            ctx.eps = eps
            return y

        @staticmethod
        def backward(ctx, dy):
            """
            Derivatives for RMSNorm:
              dx_c = (dy_c * w_c) * r - (r^3 / C) * x_c * sum_{c'}(dy_{c'} * w_{c'} * x_{c'})
              dw_c = sum(dy_c * (x_c * r))
              db_c = sum(dy_c)
            """
            x, r, w = ctx.saved_tensors
            C = x.shape[1]

            # reshape helpers
            w_b = w.view(1, -1, 1, 1)

            # common terms
            wdy = dy * w_b
            S = (wdy * x).sum(dim=1, keepdim=True)  # sum over channels

            r3_over_C = (r * r * r) / C
            dx = wdy * r - r3_over_C * x * S

            dw = (dy * (x * r)).sum(dim=(0, 2, 3))
            db = dy.sum(dim=(0, 2, 3))
            return dx, dw, db, None
