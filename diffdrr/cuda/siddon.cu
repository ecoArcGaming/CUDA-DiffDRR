#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAFunctions.h>
#define CUDA_CHECK(err)                                                        \
  do {                                                                         \
    cudaError_t err_ = (err);                                                  \
    if (err_ != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error %d at %s:%d: %s\n", err_, __FILE__,          \
              __LINE__, cudaGetErrorString(err_));                             \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

__device__ inline float get_voxel_value_trilinear_acf_dhw(
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> volume,
    const float x_grid,
    const float y_grid,
    const float z_grid,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> dims,
    const float eps);

__device__ inline void accumulate_gradient_trilinear_acf_dhw(
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> grad_volume,
    const float grad_sample_value,
    const float x_grid,
    const float y_grid,
    const float z_grid,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> dims,
    const float eps);


__global__ void compute_alphas_kernel(
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> source_acc,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> target_acc,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> dims_acc,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> alphas_acc,
    const float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int num_batches = source_acc.size(0);
    int rays_per_batch = source_acc.size(1);
    int total_rays = num_batches * rays_per_batch;

    if (idx >= total_rays) {
        return;
    }

    int b = idx / rays_per_batch;
    int r = idx % rays_per_batch;
    float sx = source_acc[b][r][0];
    float sy = source_acc[b][r][1];
    float sz = source_acc[b][r][2];
    float tx = target_acc[b][r][0];
    float ty = target_acc[b][r][1];
    float tz = target_acc[b][r][2];

    float dx = tx - sx;
    float dy = ty - sy;
    float dz = tz - sz;

    int current_alpha_idx = 0;

    for (int i = 0; i <= dims_acc[0]; ++i) {
        if (abs(dx) > eps) {
            alphas_acc[b][r][current_alpha_idx++] = (static_cast<float>(i) - sx) / dx;
        } else {
             alphas_acc[b][r][current_alpha_idx++] = copysignf(HUGE_VALF, (static_cast<float>(i) - sx));
        }
    }

    for (int i = 0; i <= dims_acc[1]; ++i) {
        if (abs(dy) > eps) {
            alphas_acc[b][r][current_alpha_idx++] = (static_cast<float>(i) - sy) / dy;
        } else {
            alphas_acc[b][r][current_alpha_idx++] = copysignf(HUGE_VALF, (static_cast<float>(i) - sy));
        }
    }

    for (int i = 0; i <= dims_acc[2]; ++i) {
        if (abs(dz) > eps) {
            alphas_acc[b][r][current_alpha_idx++] = (static_cast<float>(i) - sz) / dz;
        } else {
            alphas_acc[b][r][current_alpha_idx++] = copysignf(HUGE_VALF, (static_cast<float>(i) - sz));
        }
    }
  
}

__global__ void siddon_raycast_kernel(
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> volume_acc,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> source_acc,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> target_acc,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> sorted_alphas_acc,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> dims_acc,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> output_acc,
    const int num_alphas_per_ray,
    const float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int num_batches = source_acc.size(0);
    int rays_per_batch = source_acc.size(1);
    int total_rays = num_batches * rays_per_batch;

    if (idx >= total_rays) {
        return;
    }

    int b = idx / rays_per_batch;
    int r = idx % rays_per_batch;
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

    for (int i = 0; i < num_alphas_per_ray - 1; ++i) {
        float alpha1 = sorted_alphas_acc[b][r][i];
        float alpha2 = sorted_alphas_acc[b][r][i+1];

        if (abs(alpha1 - alpha2) < eps || !isfinite(alpha1) || !isfinite(alpha2)) {
            continue;
        }

        float alphamid = (alpha1 + alpha2) / 2.0f;

        if (alphamid < 0.0f || alphamid > 1.0f) {
            continue;
        }

        float mid_x = sx + alphamid * ray_dx;
        float mid_y = sy + alphamid * ray_dy;
        float mid_z = sz + alphamid * ray_dz;

        float x_grid_norm = 2.0f * mid_z / (dims_acc[2] + eps) - 1.0f;
        float y_grid_norm = 2.0f * mid_y / (dims_acc[1] + eps) - 1.0f;
        float z_grid_norm = 2.0f * mid_x / (dims_acc[0] + eps) - 1.0f;
        float voxel_value = get_voxel_value_trilinear_acf_dhw(
            volume_acc, x_grid_norm, y_grid_norm, z_grid_norm, dims_acc, eps
        );

        float intersection_parametric_length = alpha2 - alpha1;
        accumulated_value += voxel_value * intersection_parametric_length; 
    }
    output_acc[b][r][0] = accumulated_value;
}


std::vector<torch::Tensor> siddon_fw_cu(
    torch::Tensor volume,
    torch::Tensor source,
    torch::Tensor target,
    const float eps
) {
    auto D = volume.size(0);
    auto H = volume.size(1);
    auto W = volume.size(2);
    auto dims_vec = std::vector<int64_t>{D, H, W};
    torch::Tensor dims_tensor = torch::tensor(dims_vec, torch::dtype(torch::kInt32).device(volume.device()));

    auto batch_size = source.size(0);
    auto num_rays_per_batch = source.size(1);
    auto total_rays = batch_size * num_rays_per_batch;

    int num_alphas_per_ray = (D + 1) + (H + 1) + (W + 1);
    torch::Tensor alphas_tensor = torch::empty({batch_size, num_rays_per_batch, num_alphas_per_ray}, 
                                               volume.options());

    const int threads_per_block = 256;
    const int num_blocks = (total_rays + threads_per_block - 1) / threads_per_block;
    compute_alphas_kernel<<<num_blocks, threads_per_block, 0, at::cuda::getCurrentCUDAStream()>>>(
        source.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        target.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        dims_tensor.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        alphas_tensor.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        eps
    );
    CUDA_CHECK(cudaGetLastError());

    float* alphas_ptr = alphas_tensor.data_ptr<float>();
    thrust::for_each(thrust::cuda::par.on(at::cuda::getCurrentCUDAStream()),
                     thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(total_rays),
                     [=] __device__ (int ray_idx) {
                         int b = ray_idx / num_rays_per_batch;
                         int r = ray_idx % num_rays_per_batch;
                         float* ray_alphas_start = alphas_ptr + ray_idx * num_alphas_per_ray;
                         thrust::device_ptr<float> dev_ptr_ray_alphas_start(ray_alphas_start);
                         thrust::sort(thrust::seq, dev_ptr_ray_alphas_start, 
                                     dev_ptr_ray_alphas_start + num_alphas_per_ray);
                     });
    CUDA_CHECK(cudaGetLastError());

    torch::Tensor output_tensor = torch::empty({batch_size, num_rays_per_batch, 1}, volume.options());
    siddon_raycast_kernel<<<num_blocks, threads_per_block, 0, at::cuda::getCurrentCUDAStream()>>>(
        volume.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        source.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        target.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        alphas_tensor.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        dims_tensor.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        output_tensor.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        num_alphas_per_ray,
        eps
    );
    CUDA_CHECK(cudaGetLastError());

    return {output_tensor, alphas_tensor};
}

__global__ void siddon_bw_volume_kernel(
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> grad_output_acc,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> source_acc,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> target_acc,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> sorted_alphas_acc,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> dims_acc,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> grad_volume_acc,
    const int num_alphas_per_ray,
    const float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int num_batches = source_acc.size(0);
    int rays_per_batch = source_acc.size(1);
    int total_rays = num_batches * rays_per_batch;

    if (idx >= total_rays) {
        return;
    }

    int b = idx / rays_per_batch;
    int r = idx % rays_per_batch;

    float grad_out_ray = grad_output_acc[b][r][0];
    float sx = source_acc[b][r][0];
    float sy = source_acc[b][r][1];
    float sz = source_acc[b][r][2];

    float tx = target_acc[b][r][0];
    float ty = target_acc[b][r][1];
    float tz = target_acc[b][r][2];

    float ray_dx = tx - sx;
    float ray_dy = ty - sy;
    float ray_dz = tz - sz;

    for (int i = 0; i < num_alphas_per_ray - 1; ++i) {
        float alpha1 = sorted_alphas_acc[b][r][i];
        float alpha2 = sorted_alphas_acc[b][r][i+1];

        if (abs(alpha1 - alpha2) < eps || !isfinite(alpha1) || !isfinite(alpha2) || alpha1 >= alpha2) {
            continue;
        }

        float alphamid = (alpha1 + alpha2) / 2.0f;

        if (alphamid < 0.0f || alphamid > 1.0f) {
            continue;
        }
        
        float mid_x = sx + alphamid * ray_dx;
        float mid_y = sy + alphamid * ray_dy;
        float mid_z = sz + alphamid * ray_dz;

        float x_grid_norm = 2.0f * mid_z / (dims_acc[2] + eps) - 1.0f;
        float y_grid_norm = 2.0f * mid_y / (dims_acc[1] + eps) - 1.0f;
        float z_grid_norm = 2.0f * mid_x / (dims_acc[0] + eps) - 1.0f;

        float intersection_parametric_length = alpha2 - alpha1;
        float grad_sample_value = grad_out_ray * intersection_parametric_length;
        accumulate_gradient_trilinear_acf_dhw(
            grad_volume_acc, grad_sample_value,
            x_grid_norm, y_grid_norm, z_grid_norm,
            dims_acc, eps
        );
    }
}


torch::Tensor siddon_bw_cu(
    torch::Tensor grad_output,
    torch::Tensor volume,
    torch::Tensor source,
    torch::Tensor target,
    torch::Tensor sorted_alphas,
    const float eps
) {
    auto D = volume.size(0);
    auto H = volume.size(1);
    auto W = volume.size(2);
    auto dims_vec = std::vector<int64_t>{D, H, W};
    torch::Tensor dims_tensor = torch::tensor(dims_vec, torch::dtype(torch::kInt32).device(volume.device()));
    
    int num_alphas_per_ray = (D + 1) + (H + 1) + (W + 1);
    TORCH_CHECK(sorted_alphas.size(2) == num_alphas_per_ray, "sorted_alphas last dim mismatch");

    torch::Tensor grad_volume = torch::zeros_like(volume);

    auto batch_size = source.size(0);
    auto num_rays_per_batch = source.size(1);
    auto total_rays = batch_size * num_rays_per_batch;

    const int threads_per_block = 256;
    const int num_blocks = (total_rays + threads_per_block - 1) / threads_per_block;
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

    return grad_volume;
}


__device__ inline float get_voxel_value_trilinear_acf_dhw(
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> volume,
    const float x_grid,
    const float y_grid,
    const float z_grid,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> dims,
    const float eps) {

    float D_size = dims[0];
    float H_size = dims[1];
    float W_size = dims[2];

    float x_voxel_f = (x_grid + 1.0f) / 2.0f * W_size - 0.5f;
    float y_voxel_f = (y_grid + 1.0f) / 2.0f * H_size - 0.5f;
    float z_voxel_f = (z_grid + 1.0f) / 2.0f * D_size - 0.5f;

    int x0_w = floorf(x_voxel_f);
    int y0_h = floorf(y_voxel_f);
    int z0_d = floorf(z_voxel_f);

    float xd_frac = x_voxel_f - x0_w;
    float yd_frac = y_voxel_f - y0_h;
    float zd_frac = z_voxel_f - z0_d;

    float c[2][2][2];

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
                    c[dz_i][dy_i][dx_i] = 0.0f;
                }
            }
        }
    }
    
    float c00 = c[0][0][0] * (1.0f - xd_frac) + c[0][0][1] * xd_frac;
    float c01 = c[0][1][0] * (1.0f - xd_frac) + c[0][1][1] * xd_frac;
    float c10 = c[1][0][0] * (1.0f - xd_frac) + c[1][0][1] * xd_frac;
    float c11 = c[1][1][0] * (1.0f - xd_frac) + c[1][1][1] * xd_frac;

    float c0 = c00 * (1.0f - yd_frac) + c01 * yd_frac;
    float c1 = c10 * (1.0f - yd_frac) + c11 * yd_frac;

    float val = c0 * (1.0f - zd_frac) + c1 * zd_frac;
    
    return val;
}

__device__ inline void accumulate_gradient_trilinear_acf_dhw(
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> grad_volume,
    const float grad_sample_value,
    const float x_grid,
    const float y_grid,
    const float z_grid,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> dims,
    const float eps) {

    float D_size = dims[0];
    float H_size = dims[1];
    float W_size = dims[2];

    float x_voxel_f = (x_grid + 1.0f) / 2.0f * W_size - 0.5f;
    float y_voxel_f = (y_grid + 1.0f) / 2.0f * H_size - 0.5f;
    float z_voxel_f = (z_grid + 1.0f) / 2.0f * D_size - 0.5f;

    int x0_w = floorf(x_voxel_f);
    int y0_h = floorf(y_voxel_f);
    int z0_d = floorf(z_voxel_f);

    float xd_frac = x_voxel_f - x0_w;
    float yd_frac = y_voxel_f - y0_h;
    float zd_frac = z_voxel_f - z0_d;

    // Gradients w.r.t. c0 and c1 
    float grad_c0 = grad_sample_value * (1.0f - zd_frac);
    float grad_c1 = grad_sample_value * zd_frac;

    // Gradients w.r.t. c00, c01, c10, c11 
    float grad_c00 = grad_c0 * (1.0f - yd_frac);
    float grad_c01 = grad_c0 * yd_frac;
    float grad_c10 = grad_c1 * (1.0f - yd_frac);
    float grad_c11 = grad_c1 * yd_frac;

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
                    
                    // Derivative of trilinear interpolation w.r.t. corner c[dz_i][dy_i][dx_i]
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



