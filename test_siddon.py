import torch
import time
from diffdrr.renderers import Siddon

def test_siddon_performance():
    """Compare the runtime of the CUDA and PyTorch Siddon kernels."""
    volume = torch.randn(128, 128, 128).cuda()
    source = torch.tensor([[[100.0, 0.0, 0.0]]]).cuda()
    target = torch.tensor([[[-100.0, 0.0, 0.0]]]).cuda()
    img = torch.zeros(1, 128, 128).cuda()

    siddon_cuda = Siddon(kernel="cuda")
    siddon_pytorch = Siddon(kernel="pytorch")

    start_time = time.time()
    for _ in range(10):
        siddon_cuda(volume, source, target, img)
    torch.cuda.synchronize()
    end_time = time.time()
    cuda_time = (end_time - start_time) / 10
    print(f"CUDA kernel average runtime: {cuda_time:.4f} seconds")

    start_time = time.time()
    for _ in range(10):
        siddon_pytorch(volume, source, target, img)
    torch.cuda.synchronize()
    end_time = time.time()
    pytorch_time = (end_time - start_time) / 10
    print(f"PyTorch kernel average runtime: {pytorch_time:.4f} seconds")

if __name__ == "__main__":
    test_siddon_performance()
    # results: 
    # CUDA kernel average runtime: 0.0008 seconds                                                                                                                                             │
 │  # PyTorch kernel average runtime: 0.0037 seconds 
