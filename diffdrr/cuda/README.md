## CUDA extension (Siddon ray tracing)

This directory contains a CUDA-accelerated implementation of Siddon ray tracing used by DiffDRR for fast, differentiable DRR generation.

### What’s here
- `siddon.cu`/`siddon.cpp`: CUDA kernels and C++ wrappers for forward and backward passes.
- `siddon.py`: PyTorch JIT build + `autograd.Function` wrapper (`siddon_cuda`).

### Build and requirements
- Built JIT on import via `torch.utils.cpp_extension.load` (no manual build step).
- Requires: PyTorch with CUDA, a matching CUDA toolkit (with `nvcc`), a C++17 compiler.

### API (shapes and dtypes)
- Inputs
  - `volume`: `(D, H, W)` float32 CUDA tensor
  - `source`: `(B, N, 3)` float32 CUDA tensor (ray starts)
  - `target`: `(B, N, 3)` float32 CUDA tensor (ray ends)
- Output
  - `img`: `(B, N, 1)` float32 CUDA tensor (line integrals per ray)
- All tensors must be contiguous and on the same CUDA device.
