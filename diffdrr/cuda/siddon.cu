#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm> // For std::min, std::max

// CUDA Thrust for sorting
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h> // For thrust::device
#include <c10/cuda/CUDAStream.h> // For c10::cuda::CUDAStream
#include <c10/cuda/CUDAFunctions.h> // For c10::cuda::getCurrentCUDAStream (and others)
// CUDA error checking utility
#define CUDA_CHECK(err)                                                        \
  do {                                                                         \
    cudaError_t err_ = (err);                                                  \
    if (err_ != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error %d at %s:%d: %s\n", err_, __FILE__,          \
              __LINE__, cudaGetErrorString(err_));                             \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Forward declarations of helper device functions
__device__ inline float get_voxel_value_trilinear_acf_dhw(
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> volume,
    const float x_grid, // samples Width dimension
    const float y_grid, // samples Height dimension
    const float z_grid, // samples Depth dimension
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> dims,
    const float eps);

__device__ inline void accumulate_gradient_trilinear_acf_dhw(
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> grad_volume,
    const float grad_sample_value,
    const float x_grid, // samples Width dimension
    const float y_grid, // samples Height dimension
    const float z_grid, // samples Depth dimension
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> dims,
    const float eps);


// CUDA kernel for computing alpha intersections
__global__ void compute_alphas_kernel(
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> source_acc, // (Batch, NumRays, 3)
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> target_acc, // (Batch, NumRays, 3)
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> dims_acc,     // (3) -> {Depth, Height, Width}
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> alphas_acc, // (Batch, NumRays, MaxAlphas)
    const float eps
) {
    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int num_batches = source_acc.size(0);
    int rays_per_batch = source_acc.size(1);
    int total_rays = num_batches * rays_per_batch;

    if (idx >= total_rays) {
        return; // kernel out of bound
    }

    // Determine batch index and ray index within the batch
    int b = idx / rays_per_batch;
    int r = idx % rays_per_batch;
    float sx = source_acc[b][r][0];
    float sy = source_acc[b][r][1];
    float sz = source_acc[b][r][2];
    float tx = target_acc[b][r][0];
    float ty = target_acc[b][r][1];
    float tz = target_acc[b][r][2];

    // Ray directions
    float dx = tx - sx;
    float dy = ty - sy;
    float dz = tz - sz;

    int current_alpha_idx = 0;

    // formulas derived by Vivek
    for (int i = 0; i <= dims_acc[0]; ++i) { // x-axis
        if (abs(dx) > eps) {
            alphas_acc[b][r][current_alpha_idx++] = (static_cast<float>(i) - sx) / dx;
        } else { // Ray parallel to plane
             alphas_acc[b][r][current_alpha_idx++] = copysignf(HUGE_VALF, (static_cast<float>(i) - sx));
        }
    }

    for (int i = 0; i <= dims_acc[1]; ++i) { // y-axis
        if (abs(dy) > eps) {
            alphas_acc[b][r][current_alpha_idx++] = (static_cast<float>(i) - sy) / dy;
        } else {
            alphas_acc[b][r][current_alpha_idx++] = copysignf(HUGE_VALF, (static_cast<float>(i) - sy));
        }
    }

    for (int i = 0; i <= dims_acc[2]; ++i) { // z-axis
        if (abs(dz) > eps) {
            alphas_acc[b][r][current_alpha_idx++] = (static_cast<float>(i) - sz) / dz;
        } else {
            alphas_acc[b][r][current_alpha_idx++] = copysignf(HUGE_VALF, (static_cast<float>(i) - sz));
        }
    }
  
}

// CUDA kernel for computing midpoints and sampling volume
__global__ void siddon_raycast_kernel(
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> volume_acc,    // (Depth, Height, Width)
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> source_acc,    // (Batch, NumRays, 3)
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> target_acc,    // (Batch, NumRays, 3)
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> sorted_alphas_acc, // (Batch, NumRays, MaxAlphas)
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> dims_acc,        // (3) -> {Depth, Height, Width}
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> output_acc,    // (Batch, NumRays, 1)
    const int num_alphas_per_ray, // Actual number of alphas computed per ray ( D+1 + H+1 + W+1 )
    const float eps
) {
    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int num_batches = source_acc.size(0);
    int rays_per_batch = source_acc.size(1);
    int total_rays = num_batches * rays_per_batch;

    if (idx >= total_rays) {
        return;
    }

    // Determine batch index and ray index within the batch
    int b = idx / rays_per_batch;
    int r = idx % rays_per_batch;

    // Source and target coordinates for the current ray
    float sx = source_acc[b][r][0];
    float sy = source_acc[b][r][1];
    float sz = source_acc[b][r][2];

    float tx = target_acc[b][r][0];
    float ty = target_acc[b][r][1];
    float tz = target_acc[b][r][2];
    
    float ray_dx = tx - sx;
    float ray_dy = ty - sy;
    float ray_dz = tz - sz;

    float accumulated_value = 0.0f;

    // Iterate through pairs of sorted alphas to find segments
    for (int i = 0; i < num_alphas_per_ray - 1; ++i) {
        float alpha1 = sorted_alphas_acc[b][r][i];
        float alpha2 = sorted_alphas_acc[b][r][i+1];

        // Skip if alphas are identical (no segment) or invalid (e.g. inf)
        if (abs(alpha1 - alpha2) < eps || !isfinite(alpha1) || !isfinite(alpha2)) {
            continue;
        }

        float alphamid = (alpha1 + alpha2) / 2.0f;

        // out of volume, discard
        if (alphamid < 0.0f || alphamid > 1.0f) {
            continue;
        }

        // in world space
        float mid_x = sx + alphamid * ray_dx;
        float mid_y = sy + alphamid * ray_dy;
        float mid_z = sz + alphamid * ray_dz;

        float x_grid_norm = 2.0f * mid_z / (dims_acc[2] + eps) - 1.0f; // samples Width
        float y_grid_norm = 2.0f * mid_y / (dims_acc[1] + eps) - 1.0f; // samples Height
        float z_grid_norm = 2.0f * mid_x / (dims_acc[0] + eps) - 1.0f; // samples Depth
        
        // Sample volume using trilinear interpolation
        float voxel_value = get_voxel_value_trilinear_acf_dhw(
            volume_acc, x_grid_norm, y_grid_norm, z_grid_norm, dims_acc, eps
        );

        float intersection_parametric_length = alpha2 - alpha1;
        accumulated_value += voxel_value * intersection_parametric_length; 
    }
    output_acc[b][r][0] = accumulated_value;
}


// Forward pass CUDA implementation
std::vector<torch::Tensor> siddon_fw_cu(
    torch::Tensor volume,      // (D, H, W)
    torch::Tensor source,      // (B, N, 3)
    torch::Tensor target,      // (B, N, 3)
    const float eps
) {
   
    // Volume dimensions
    auto D = volume.size(0);
    auto H = volume.size(1);
    auto W = volume.size(2);
    auto dims_vec = std::vector<int64_t>{D, H, W};
    torch::Tensor dims_tensor = torch::tensor(dims_vec, torch::dtype(torch::kInt32).device(volume.device()));

    // Batch and ray dimensions
    auto batch_size = source.size(0);
    auto num_rays_per_batch = source.size(1);
    auto total_rays = batch_size * num_rays_per_batch;

    // Max number of alpha intersections per ray, sum of 3 axes
    int num_alphas_per_ray = (D + 1) + (H + 1) + (W + 1);

    // Allocate alphas tensor: (Batch, NumRays, MaxAlphas)
    torch::Tensor alphas_tensor = torch::empty({batch_size, num_rays_per_batch, num_alphas_per_ray}, 
                                               volume.options());

    // Kernel launch parameters
    const int threads_per_block = 256;
    const int num_blocks = (total_rays + threads_per_block - 1) / threads_per_block;

    // Call compute_alphas_kernel
    compute_alphas_kernel<<<num_blocks, threads_per_block, 0, at::cuda::getCurrentCUDAStream()>>>(
        source.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        target.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        dims_tensor.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        alphas_tensor.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        eps
    );
    CUDA_CHECK(cudaGetLastError());

    // Sort alphas for each ray using Thrust (in-place)
    float* alphas_ptr = alphas_tensor.data_ptr<float>();
    for (int b = 0; b < batch_size; ++b) {
        for (int r = 0; r < num_rays_per_batch; ++r) {
            float* ray_alphas_start = alphas_ptr + (b * num_rays_per_batch + r) * num_alphas_per_ray;
            thrust::device_ptr<float> dev_ptr_ray_alphas_start(ray_alphas_start);
            // Using thrust::device execution policy with current stream
            thrust::sort(thrust::cuda::par.on(at::cuda::getCurrentCUDAStream()), 
                         dev_ptr_ray_alphas_start, 
                         dev_ptr_ray_alphas_start + num_alphas_per_ray);
        }
    }
    CUDA_CHECK(cudaGetLastError()); // Check for errors after Thrust operations

    // Allocate output tensor: (Batch, NumRays, 1)
    torch::Tensor output_tensor = torch::empty({batch_size, num_rays_per_batch, 1}, volume.options());

    // Call siddon_raycast_kernel
    siddon_raycast_kernel<<<num_blocks, threads_per_block, 0, at::cuda::getCurrentCUDAStream()>>>(
        volume.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        source.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        target.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        alphas_tensor.packed_accessor32<float, 3, torch::RestrictPtrTraits>(), // alphas_tensor is now sorted
        dims_tensor.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        output_tensor.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        num_alphas_per_ray,
        eps
    );
    CUDA_CHECK(cudaGetLastError());

    return {output_tensor, alphas_tensor};
}

// Backward pass kernel for volume gradients
__global__ void siddon_bw_volume_kernel(
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> grad_output_acc, // (Batch, NumRays, 1)
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> source_acc,      // (Batch, NumRays, 3)
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> target_acc,      // (Batch, NumRays, 3)
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> sorted_alphas_acc, // (Batch, NumRays, MaxAlphas)
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> dims_acc,          // (3) -> {Depth, Height, Width}
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> grad_volume_acc, // (Depth, Height, Width)
    const int num_alphas_per_ray,
    const float eps
) {
    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int num_batches = source_acc.size(0);
    int rays_per_batch = source_acc.size(1);
    int total_rays = num_batches * rays_per_batch;

    if (idx >= total_rays) {
        return;
    }

    // Determine batch index and ray index within the batch
    int b = idx / rays_per_batch;
    int r = idx % rays_per_batch;

    // Gradient for the current ray's output
    float grad_out_ray = grad_output_acc[b][r][0];

    // Source and target coordinates for the current ray
    float sx = source_acc[b][r][0];
    float sy = source_acc[b][r][1];
    float sz = source_acc[b][r][2];

    float tx = target_acc[b][r][0];
    float ty = target_acc[b][r][1];
    float tz = target_acc[b][r][2];

    float ray_dx = tx - sx;
    float ray_dy = ty - sy;
    float ray_dz = tz - sz;

    // Iterate through pairs of sorted alphas (segments)
    for (int i = 0; i < num_alphas_per_ray - 1; ++i) {
        float alpha1 = sorted_alphas_acc[b][r][i];
        float alpha2 = sorted_alphas_acc[b][r][i+1];

        if (abs(alpha1 - alpha2) < eps || !isfinite(alpha1) || !isfinite(alpha2)) {
            continue;
        }
        if (alpha1 >= alpha2) continue;

        float alphamid = (alpha1 + alpha2) / 2.0f;

        if (alphamid < 0.0f || alphamid > 1.0f) {
            continue;
        }
        
        // Calculate midpoint XYZ coordinates in world space
        float mid_x = sx + alphamid * ray_dx;
        float mid_y = sy + alphamid * ray_dy;
        float mid_z = sz + alphamid * ray_dz;

        // Normalize coordinates for grid sampling
        float x_grid_norm = 2.0f * mid_z / (dims_acc[2] + eps) - 1.0f; // samples Width
        float y_grid_norm = 2.0f * mid_y / (dims_acc[1] + eps) - 1.0f; // samples Height
        float z_grid_norm = 2.0f * mid_x / (dims_acc[0] + eps) - 1.0f; // samples Depth

        // parametric length
        float intersection_parametric_length = alpha2 - alpha1;

        // Grad of output w.r.t. this specific voxel_value sample
        float grad_sample_value = grad_out_ray * intersection_parametric_length;
        
        // Accumulate gradient to volume using adjoint of trilinear interpolation
        accumulate_gradient_trilinear_acf_dhw(
            grad_volume_acc, grad_sample_value,
            x_grid_norm, y_grid_norm, z_grid_norm,
            dims_acc, eps
        );
    }
}


// Backward pass CUDA implementation
torch::Tensor siddon_bw_cu(
    torch::Tensor grad_output, // (B, N, 1)
    torch::Tensor volume,      // (D, H, W) - needed for dims and device
    torch::Tensor source,      // (B, N, 3) - needed for recomputing alphas or passing sorted
    torch::Tensor target,      // (B, N, 3)
    torch::Tensor sorted_alphas, // (B, N, MaxAlphas) - Pass from forward or recompute
    const float eps
) {
   

    // Volume dimensions
    auto D = volume.size(0);
    auto H = volume.size(1);
    auto W = volume.size(2);
    auto dims_vec = std::vector<int64_t>{D, H, W};
    torch::Tensor dims_tensor = torch::tensor(dims_vec, torch::dtype(torch::kInt32).device(volume.device()));
    
    int num_alphas_per_ray = (D + 1) + (H + 1) + (W + 1);
    TORCH_CHECK(sorted_alphas.size(2) == num_alphas_per_ray, "sorted_alphas last dim mismatch");

    //  grad_volume tensor
    torch::Tensor grad_volume = torch::zeros_like(volume);

    auto batch_size = source.size(0);
    auto num_rays_per_batch = source.size(1);
    auto total_rays = batch_size * num_rays_per_batch;

    // Kernel launch parameters
    const int threads_per_block = 256;
    const int num_blocks = (total_rays + threads_per_block - 1) / threads_per_block;

    // Call siddon_bw_volume_kernel
    siddon_bw_volume_kernel<<<num_blocks, threads_per_block, 0, at::cuda::getCurrentCUDAStream()>>>(
        grad_output.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        source.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        target.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        sorted_alphas.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        dims_tensor.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        grad_volume.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        num_alphas_per_ray,
        eps
    );
    CUDA_CHECK(cudaGetLastError());

    // Gradients for source and target are not computed here.
    return grad_volume; // grad_volume, grad_source (nullptr), grad_target (nullptr)
}


// Helper device function for trilinear interpolation (align_corners=False)
// Grid coords (x_grid, y_grid, z_grid) sample Width, Height, Depth dimensions respectively.
// Volume is indexed as volume[depth_idx][height_idx][width_idx].
// dims are [Depth, Height, Width].
__device__ inline float get_voxel_value_trilinear_acf_dhw(
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> volume,
    const float x_grid, // samples Width dimension
    const float y_grid, // samples Height dimension
    const float z_grid, // samples Depth dimension
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> dims,
    const float eps) {

    float D_size = dims[0]; // Depth
    float H_size = dims[1]; // Height
    float W_size = dims[2]; // Width

    // Unnormalize grid coordinates and shift to voxel centers
    // x_voxel_f samples Width dimension, range [-0.5, W_size - 0.5]
    float x_voxel_f = (x_grid + 1.0f) / 2.0f * W_size - 0.5f;
    float y_voxel_f = (y_grid + 1.0f) / 2.0f * H_size - 0.5f;
    float z_voxel_f = (z_grid + 1.0f) / 2.0f * D_size - 0.5f;

    int x0_w = floorf(x_voxel_f); // Width index
    int y0_h = floorf(y_voxel_f); // Height index
    int z0_d = floorf(z_voxel_f); // Depth index

    // Fractional parts
    float xd_frac = x_voxel_f - x0_w; 
    float yd_frac = y_voxel_f - y0_h; 
    float zd_frac = z_voxel_f - z0_d; 

    float c[2][2][2]; // c[dz][dy][dx]

    for (int dz_i = 0; dz_i < 2; ++dz_i) {
        for (int dy_i = 0; dy_i < 2; ++dy_i) {
            for (int dx_i = 0; dx_i < 2; ++dx_i) {
                int current_d = z0_d + dz_i;
                int current_h = y0_h + dy_i;
                int current_w = x0_w + dx_i;
                if (current_d >= 0 && current_d < D_size &&
                    current_h >= 0 && current_h < H_size &&
                    current_w >= 0 && current_w < W_size) {
                    c[dz_i][dy_i][dx_i] = volume[current_d][current_h][current_w];
                } else {
                    c[dz_i][dy_i][dx_i] = 0.0f; // Boundary condition: zero padding
                }
            }
        }
    }
    
    // Interpolate along x (Width)
    float c00 = c[0][0][0] * (1.0f - xd_frac) + c[0][0][1] * xd_frac;
    float c01 = c[0][1][0] * (1.0f - xd_frac) + c[0][1][1] * xd_frac;
    float c10 = c[1][0][0] * (1.0f - xd_frac) + c[1][0][1] * xd_frac;
    float c11 = c[1][1][0] * (1.0f - xd_frac) + c[1][1][1] * xd_frac;

    // Interpolate along y (Height)
    float c0 = c00 * (1.0f - yd_frac) + c01 * yd_frac;
    float c1 = c10 * (1.0f - yd_frac) + c11 * yd_frac;

    // Interpolate along z (Depth)
    float val = c0 * (1.0f - zd_frac) + c1 * zd_frac;
    
    return val;
}

// Helper device function for accumulating gradients for trilinear interpolation
__device__ inline void accumulate_gradient_trilinear_acf_dhw(
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> grad_volume,
    const float grad_sample_value,
    const float x_grid, // samples Width dimension
    const float y_grid, // samples Height dimension
    const float z_grid, // samples Depth dimension
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> dims,
    const float eps) {

    float D_size = dims[0]; // Depth
    float H_size = dims[1]; // Height
    float W_size = dims[2]; // Width

    float x_voxel_f = (x_grid + 1.0f) / 2.0f * W_size - 0.5f;
    float y_voxel_f = (y_grid + 1.0f) / 2.0f * H_size - 0.5f;
    float z_voxel_f = (z_grid + 1.0f) / 2.0f * D_size - 0.5f;

    int x0_w = floorf(x_voxel_f);
    int y0_h = floorf(y_voxel_f);
    int z0_d = floorf(z_voxel_f);

    float xd_frac = x_voxel_f - x0_w;
    float yd_frac = y_voxel_f - y0_h;
    float zd_frac = z_voxel_f - z0_d;

    // Gradients w.r.t. c0 and c1 (intermediate values in z-interpolation)
    float grad_c0 = grad_sample_value * (1.0f - zd_frac);
    float grad_c1 = grad_sample_value * zd_frac;

    // Gradients w.r.t. c00, c01, c10, c11 (intermediate values in y-interpolation)
    float grad_c00 = grad_c0 * (1.0f - yd_frac);
    float grad_c01 = grad_c0 * yd_frac;
    float grad_c10 = grad_c1 * (1.0f - yd_frac);
    float grad_c11 = grad_c1 * yd_frac;

    // Loop over the 8 corner voxels
    for (int dz_i = 0; dz_i < 2; ++dz_i) { // Corresponds to z0_d, z0_d+1
        for (int dy_i = 0; dy_i < 2; ++dy_i) { // Corresponds to y0_h, y0_h+1
            for (int dx_i = 0; dx_i < 2; ++dx_i) { // Corresponds to x0_w, x0_w+1
                int current_d = z0_d + dz_i;
                int current_h = y0_h + dy_i;
                int current_w = x0_w + dx_i;

                if (current_d >= 0 && current_d < D_size &&
                    current_h >= 0 && current_h < H_size &&
                    current_w >= 0 && current_w < W_size) {
                    
                    float w_dx = (dx_i == 0) ? (1.0f - xd_frac) : xd_frac;
                    float w_dy = (dy_i == 0) ? (1.0f - yd_frac) : yd_frac;
                    float w_dz = (dz_i == 0) ? (1.0f - zd_frac) : zd_frac;
                    
                    // This is the gradient of the output value w.r.t specific corner c[dz_i][dy_i][dx_i]
                    // Need to chain rule from grad_c00, etc.
                    float grad_corner_val;
                    if (dz_i == 0) { // affects c0
                        if (dy_i == 0) { // affects c00
                            grad_corner_val = grad_c00 * ((dx_i == 0) ? (1.0f - xd_frac) : xd_frac);
                        } else { // affects c01
                            grad_corner_val = grad_c01 * ((dx_i == 0) ? (1.0f - xd_frac) : xd_frac);
                        }
                    } else { // affects c1
                        if (dy_i == 0) { // affects c10
                            grad_corner_val = grad_c10 * ((dx_i == 0) ? (1.0f - xd_frac) : xd_frac);
                        } else { // affects c11
                            grad_corner_val = grad_c11 * ((dx_i == 0) ? (1.0f - xd_frac) : xd_frac);
                        }
                    }
                    // The above logic for grad_corner_val is incorrect. Simpler:
                    // Derivative of `val` w.r.t. `c[dz_i][dy_i][dx_i]`
                    float weight = w_dx * w_dy * w_dz; // This is not correct either
                    
                    // Correct weights for gradient distribution:
                    // d(val)/d(c[0][0][0]) = (1-zd_frac)(1-yd_frac)(1-xd_frac)
                    // d(val)/d(c[0][0][1]) = (1-zd_frac)(1-yd_frac)(xd_frac)
                    // ... and so on for all 8 corners.
                    float grad_contrib_to_corner = grad_sample_value *
                                                   ((dz_i == 0) ? (1.0f - zd_frac) : zd_frac) *
                                                   ((dy_i == 0) ? (1.0f - yd_frac) : yd_frac) *
                                                   ((dx_i == 0) ? (1.0f - xd_frac) : xd_frac);

                    atomicAdd(&(grad_volume[current_d][current_h][current_w]), grad_contrib_to_corner);
                }
            }
        }
    }
}

// Pybind11 module definition (example, if you were to build this as an extension)
// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("siddon_forward", &siddon_fw_cu, "Siddon forward (CUDA)");
//   m.def("siddon_backward", &siddon_bw_cu, "Siddon backward (CUDA)");
// }

