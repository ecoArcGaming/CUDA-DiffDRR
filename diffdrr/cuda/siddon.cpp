#include <torch/extension.h>
#include <iostream>
#include <vector>

// Must be valid for CUDA tensors
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Load CUDA functions
std::vector<torch::Tensor> siddon_fw_cu(
    torch::Tensor volume, 
    torch::Tensor source, 
    torch::Tensor target,
    const float eps
);

torch::Tensor siddon_bw_cu(
    torch::Tensor grad_output, // (B, N, 1)
    torch::Tensor volume,      // (D, H, W) - needed for dims and device
    torch::Tensor source,      // (B, N, 3) - needed for recomputing alphas or passing sorted
    torch::Tensor target,      // (B, N, 3)
    torch::Tensor sorted_alphas, // (B, N, MaxAlphas) - Pass from forward or recompute
    const float eps

);

// Forward pass wrapper
std::vector<torch::Tensor> fw(
    torch::Tensor volume, 
    torch::Tensor source, 
    torch::Tensor target
) {
    CHECK_INPUT(volume);
    CHECK_INPUT(source);
    CHECK_INPUT(target);
    
    // Ensure tensors are float32
    volume = volume.to(torch::kFloat32);
    source = source.to(torch::kFloat32);
    target = target.to(torch::kFloat32);
    const float eps = 1e-8;
    return siddon_fw_cu(volume, source, target, eps);
}

// Backward pass wrapper
torch::Tensor bw(
    torch::Tensor grad_output,
    torch::Tensor volume,
    torch::Tensor source,
    torch::Tensor target,
    torch::Tensor alphas
) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(volume);
    CHECK_INPUT(source);
    CHECK_INPUT(target);
    
    // Ensure tensors are float32
    grad_output = grad_output.to(torch::kFloat32);
    volume = volume.to(torch::kFloat32);
    source = source.to(torch::kFloat32);
    target = target.to(torch::kFloat32);
    const float eps = 1e-8;

    return siddon_bw_cu(grad_output, volume, source, target, alphas, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("siddon_fw", &fw, "Siddon forward pass");
    m.def("siddon_bw", &bw, "Siddon backward pass");
}