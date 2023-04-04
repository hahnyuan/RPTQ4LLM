#include <ATen/ATen.h>
#include <cuda.h>
#include <torch/extension.h>
#include <cuda_fp16.h>

/*
Utilize shared memory for input data:
Loading input data into shared memory can help reduce global memory accesses, as shared memory has much lower latency than global memory.

Use 2D grid and 2D block:
The current blockDim and gridDim configuration might not efficiently utilize the GPU hardware. Consider using a 2D grid and 2D block for better occupancy.

Minimize shared memory bank conflicts:
To avoid shared memory bank conflicts, pad shared memory to ensure that consecutive threads access different banks.

NOTE: optmize not ok
*/
__global__ void layer_norm_kernel_fp16_optimized(const at::Half *input, at::Half *output, at::Half *scale, at::Half *shift, long *dst_index, int b, int c) {
    constexpr int TILE_SIZE = 32;
    int idx = blockIdx.x * TILE_SIZE + threadIdx.x;
    int idy = blockIdx.y * TILE_SIZE + threadIdx.y;

    __shared__ half shared_input[TILE_SIZE][TILE_SIZE + 1];
    __shared__ half shared_mean[TILE_SIZE];
    __shared__ half shared_var[TILE_SIZE];

    if (idx < b && idy < c) {
        shared_input[threadIdx.x][threadIdx.y] = input[idx * c + idy];
    } else {
        shared_input[threadIdx.x][threadIdx.y] = __float2half(0.0f);
    }
    __syncthreads();

    // Calculate mean using parallel reduction
    half thread_sum = __float2half(0.0f);
    for (int i = threadIdx.y; i < TILE_SIZE; i += blockDim.y) {
        thread_sum = __hadd(thread_sum, shared_input[threadIdx.x][i]);
    }
    shared_mean[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int stride = TILE_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.y < stride) {
            shared_mean[threadIdx.x] = __hadd(shared_mean[threadIdx.x], shared_mean[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.y == 0) {
        shared_mean[threadIdx.x] = __hdiv(shared_mean[threadIdx.x], __int2half_rn(c));
    }
    __syncthreads();

    half mean = shared_mean[threadIdx.x];

    // Calculate variance using parallel reduction
    half thread_var_sum = __float2half(0.0f);
    for (int i = threadIdx.y; i < TILE_SIZE; i += blockDim.y) {
        half diff = __hsub(shared_input[threadIdx.x][i], mean);
        thread_var_sum = __hadd(thread_var_sum, __hmul(diff, diff));
    }
    shared_var[threadIdx.x] = thread_var_sum;
    __syncthreads();

    for (int stride = TILE_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.y < stride) {
            shared_var[threadIdx.x] = __hadd(shared_var[threadIdx.x], shared_var[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.y == 0) {
        shared_var[threadIdx.x] = __hdiv(shared_var[threadIdx.x], __int2half_rn(c));
    }
    __syncthreads();

    half var = shared_var[threadIdx.x];

    // Normalize
    if (idx < b && idy < c) {
        half normalized_value = __hdiv(__hsub(shared_input[threadIdx.x][threadIdx.y], mean), hsqrt(__hadd(var, __float2half(1e-5f))));
        int dst_linear_idy = idx * c + dst_index[idy];
        output[dst_linear_idy] = __hfma(normalized_value, scale[idy], shift[idy]);
    }
}


__global__ void layer_norm_kernel_fp16(const at::Half *input, at::Half *output, at::Half *scale, at::Half *shift, long *dst_index, int b, int c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < b && idy < c) {
        // Calculate mean using parallel reduction
        __shared__ half shared_mean[32][33];
        half thread_sum = __float2half(0.0f);
        for (int i = threadIdx.y; i < c; i += blockDim.y) {
            int linear_idx = idx * c + i;
            thread_sum = __hadd(thread_sum, input[linear_idx]);
        }
        shared_mean[threadIdx.x][threadIdx.y] = thread_sum;
        __syncthreads();


        for (int stride = blockDim.y / 2; stride > 0; stride >>= 1) {
            if (threadIdx.y < stride) {
                shared_mean[threadIdx.x][threadIdx.y] = __hadd(shared_mean[threadIdx.x][threadIdx.y], shared_mean[threadIdx.x][threadIdx.y + stride]);
            }
            __syncthreads();
        }

        if (threadIdx.y == 0) {
            shared_mean[threadIdx.x][0] = __hdiv(shared_mean[threadIdx.x][0], __int2half_rn(c));
        }
        __syncthreads();

        half mean = shared_mean[threadIdx.x][0];

        // Calculate variance using parallel reduction
        __shared__ half shared_var[32][33];
        half thread_var_sum = __float2half(0.0f);
        for (int i = threadIdx.y; i < c; i += blockDim.y) {
            int linear_idx = idx * c + i;
            half diff = __hsub(input[linear_idx], mean);
            thread_var_sum = __hadd(thread_var_sum, __hmul(diff, diff));
        }
        shared_var[threadIdx.x][threadIdx.y] = thread_var_sum;
        __syncthreads();

        for (int stride = blockDim.y / 2; stride > 0; stride >>= 1) {
            if (threadIdx.y < stride) {
                shared_var[threadIdx.x][threadIdx.y] = __hadd(shared_var[threadIdx.x][threadIdx.y], shared_var[threadIdx.x][threadIdx.y + stride]);
            }
            __syncthreads();
        }

        if (threadIdx.y == 0) {
            shared_var[threadIdx.x][0] = __hdiv(shared_var[threadIdx.x][0], __int2half_rn(c));
        }
        __syncthreads();

        half var = shared_var[threadIdx.x][0];

        // Normalize input
        int linear_idx = idx * c + idy;
        half normalized_value = __hdiv(__hsub(input[linear_idx], mean), hsqrt(__hadd(var, __float2half(1e-5f))));
        int dst_linear_idy = idx * c + dst_index[idy];
        output[dst_linear_idy] = __hfma(normalized_value, scale[idy], shift[idy]);
    }
}


void reorder_layer_norm_fp16(at::Tensor input, at::Tensor output, at::Tensor scale, at::Tensor shift, at::Tensor dst_index) {
    int b = input.size(0);
    int c = input.size(1);

    dim3 blockDim(32, 32);
    dim3 gridDim((b + blockDim.x - 1) / blockDim.x, (c + blockDim.y - 1) / blockDim.y);

    layer_norm_kernel_fp16<<<gridDim, blockDim>>>(
        input.data_ptr<at::Half>(),
        output.data_ptr<at::Half>(),
        scale.data_ptr<at::Half>(),
        shift.data_ptr<at::Half>(),
        dst_index.data_ptr<long>(),
        b, c
        );
    // layer_norm_kernel_fp16_optimized<<<gridDim, blockDim>>>(
    //     input.data_ptr<at::Half>(),
    //     output.data_ptr<at::Half>(),
    //     scale.data_ptr<at::Half>(),
    //     shift.data_ptr<at::Half>(),
    //     dst_index.data_ptr<long>(),
    //     b, c
    //     );
    cudaDeviceSynchronize();
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("forward", &layer_norm, "layer_norm operation (CUDA)");
  m.def("forward", &reorder_layer_norm_fp16, "layer_norm_fp16 operation (CUDA)");
}
