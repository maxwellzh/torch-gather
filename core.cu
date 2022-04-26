#include "core.h"

#include <stdio.h>
#include <assert.h>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <device_atomic_functions.h>
#include <device_launch_parameters.h>

#define W 64
#define H 16

__global__ void kernel_cat(const ushort *x_padded, const unsigned int *lx,
                           ushort *x_gather, const unsigned int *memPref,
                           const unsigned int N_stride, const unsigned int T_stride,
                           unsigned int V)
{
    unsigned int t = blockIdx.x * 4 + threadIdx.x;
    unsigned int n = blockIdx.z;
    if (t >= lx[n])
        return;

    unsigned int v = blockIdx.y * 256 + threadIdx.y;
    if (v >= V)
        return;

    memcpy(x_gather + memPref[n] + t * V + v, x_padded + n * N_stride + t * T_stride + v, 2);
    // x_gather[memPref[n] + t * V + v] = x_padded[n * N_stride + t * T_stride + v];
}

gatherStatus_t run_gather_cat(cudaStream_t stream, const ushort *x_padded, const unsigned int *lx,
                              ushort *x_gather, const unsigned int *memPref,
                              const unsigned int N_stride, const unsigned int T_stride,
                              unsigned int N, unsigned int T, unsigned int V)
{

    dim3 threads(4, 256);
    dim3 blocks((T + 3) / 4, (V + 255) / 256, N);

    kernel_cat<<<blocks, threads, 0, stream>>>(x_padded, lx, x_gather, memPref, N_stride, T_stride, V);
    if (cudaGetLastError() != cudaSuccess)
        return GATHER_STATUS_FAILED;

    return GATHER_STATUS_SUCCESS;
}

__global__ void kernel_pad(const ushort *grad_gather, const unsigned int *lx,
                           ushort *grad_padded, const unsigned int *memPref,
                           const unsigned int N_stride, const unsigned int T_stride,
                           unsigned int V)
{
    unsigned int t = blockIdx.x * 4 + threadIdx.x;
    unsigned int n = blockIdx.z;
    if (t >= lx[n])
        return;

    unsigned int v = blockIdx.y * 256 + threadIdx.y;
    if (v >= V)
        return;

    memcpy(grad_padded + n * N_stride + t * T_stride + v, grad_gather + memPref[n] + t * V + v, 2);
    // grad_padded[n * N_stride + t * T_stride + v] = grad_gather[memPref[n] + t * V + v];
}

gatherStatus_t run_pad_grad(cudaStream_t stream, const ushort *grad_gather, const unsigned int *lx,
                            ushort *grad_padded, const unsigned int *memPref,
                            const unsigned int N_stride, const unsigned int T_stride,
                            unsigned int N, unsigned int T, unsigned int V)
{

    dim3 threads(4, 256);
    dim3 blocks((T + 3) / 4, (V + 255) / 256, N);

    kernel_pad<<<blocks, threads, 0, stream>>>(grad_gather, lx, grad_padded, memPref, N_stride, T_stride, V);
    if (cudaGetLastError() != cudaSuccess)
        return GATHER_STATUS_FAILED;

    return GATHER_STATUS_SUCCESS;
}