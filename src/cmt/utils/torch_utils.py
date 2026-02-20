import torch
from contextlib import nullcontext

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def dtype_from_str(s: str | None) -> torch.dtype:
    s = (s or "").lower()
    if s in ("fp16", "float16", "half"): return torch.float16
    if s in ("bf16", "bfloat16"): return torch.bfloat16
    return torch.float32

def get_amp_context(enabled: bool, dtype: torch.dtype = torch.float16):
    if enabled and torch.cuda.is_available():
        return torch.amp.autocast("cuda", dtype=dtype)
    return nullcontext()

def l2_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)
