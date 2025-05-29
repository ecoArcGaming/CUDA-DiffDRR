import torch
from torch.utils.cpp_extension import load

# build CUDA extension JIT
siddon_cpp = load(name = 'siddon_cpp', 
                 sources = ['./diffdrr/cuda/siddon.cpp', './diffdrr/cuda/siddon.cu'], 
                 verbose=True)

# wrapper for gradient computations
class siddon_cuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, source, target):
        img, alphas = siddon_cpp.siddon_fw(volume.contiguous(), source.contiguous(), target.contiguous())

        ctx.save_for_backward(volume, source, target, alphas)

        return img

    @staticmethod
    def backward(ctx, grad):
        volume, source, target, alphas = ctx.saved_tensors

        dV = siddon_cpp.siddon_bw(grad.contiguous(), volume, source, target, alphas)

        return dV, None


