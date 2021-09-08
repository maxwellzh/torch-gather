#include "core.h"

#include <stdio.h>
#include <assert.h>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <device_atomic_functions.h>
#include <device_launch_parameters.h>

#define W 64
#define H 16

__global__ void kernel_fill_gather(const float *xs, const float *ys, const unsigned int *lx, const unsigned int *ly,
                                   float *gather_xs, const unsigned int *memPref,
                                   const unsigned int *framePref, const unsigned int *labelPref,
                                   unsigned int V)
{
    unsigned int t = blockIdx.x * W + threadIdx.x;
    unsigned int u = blockIdx.y * H + threadIdx.y;
    unsigned int n = blockIdx.z;

    unsigned int actual_t = lx[n];
    unsigned int actual_u = ly[n];

    if (t >= actual_t || u >= actual_u)
        return;

    float *ptr_gather = gather_xs + memPref[n] + (t * actual_u + u) * V;
    const float *ptr_x = xs + framePref[n] + t * V;
    const float *ptr_y = ys + labelPref[n] + u * V;

    for (int i = 0; i < V; i++, ptr_gather++, ptr_x++, ptr_y++)
    {
        *ptr_gather = *ptr_x + *ptr_y;
    }
}

gatherStatus_t run_gather_sum(cudaStream_t stream, const float *xs, const float *ys, const unsigned int *lx, const unsigned int *ly,
                              float *gather_xs, const unsigned int *memPref,
                              const unsigned int *framePref, const unsigned int *labelPref,
                              unsigned int N, unsigned int T, unsigned int U, unsigned int V)
{

    dim3 threads1(W, H);
    dim3 blocks1((T + W - 1) / W, (U + H - 1) / H, N);

    kernel_fill_gather<<<blocks1, threads1, 0, stream>>>(xs, ys, lx, ly, gather_xs, memPref, framePref, labelPref, V);
    if (cudaGetLastError() != cudaSuccess)
        return GATHER_STATUS_FAILED;

    return GATHER_STATUS_SUCCESS;
}

__global__ void kernel_fill_grad_x(const float *grad_sum, const unsigned int *ly,
                                   float *grad_x, const unsigned int *memPref,
                                   const unsigned int *xCumSum, unsigned int V)
{
    unsigned int v = blockIdx.y * H + threadIdx.y;
    if (v >= V)
        return;

    unsigned int xi = blockIdx.x * W + threadIdx.x;
    unsigned int n = blockIdx.z;
    unsigned int xPref = 0;
    if (n > 0)
    {
        xPref = xCumSum[n - 1];
        if (xi >= xCumSum[n] - xPref)
            return;
    }
    else if (xi >= xCumSum[0])
    {
        return;
    }
    // printf("(n, xi, v)=(%d, %d, %d)\n", n, xi, v);
    const float *ptr_grad_sum = grad_sum + memPref[n] + xi * ly[n] * V + v;
    float *ptr_x = grad_x + (xPref + xi) * V + v;
    *ptr_x = 0.0f;

    /**
     * native summation, might cause higher numerical error
     */
    // for (int i = 0; i < ly[n]; i++, ptr_grad_sum++)
    // {
    //     *ptr_x += *ptr_grad_sum;
    // }

    /**
     * below is the Kahan summation: https://en.wikipedia.org/wiki/Kahan_summation_algorithm
     * in my testing, two algorithm run in close speed, while Kahan algorithm has better presicion.
     */
    float c = 0.0f;
    float _y, _t;
    for (int i = 0; i < ly[n]; i++, ptr_grad_sum += V)
    {
        _y = *ptr_grad_sum - c;
        _t = *ptr_x + _y;
        c = (_t - *ptr_x) - _y;
        *ptr_x = _t;
    }
}

__global__ void kernel_fill_grad_y(const float *grad_sum, const unsigned int *lx,
                                   const unsigned int *ly, float *grad_y, const unsigned int *memPref,
                                   const unsigned int *yCumSum, unsigned int V)
{
    unsigned int v = blockIdx.y * H + threadIdx.y;
    if (v >= V)
        return;

    unsigned int yi = blockIdx.x * W + threadIdx.x;
    unsigned int n = blockIdx.z;
    unsigned int curN = ly[n];
    if (yi >= curN)
        return;

    unsigned int yPref = yCumSum[n] - curN;
    unsigned int _step = curN * V;

    const float *ptr_grad_sum = grad_sum + memPref[n] + yi * V + v;
    float *ptr_y = grad_y + (yPref + yi) * V + v;
    *ptr_y = 0.0f;

    /**
     * refer to kernel_fill_grad_x() for details
     */
    float c = 0.0f;
    float _y, _t;
    for (int i = 0; i < lx[n]; i++, ptr_grad_sum += _step)
    {
        _y = *ptr_grad_sum - c;
        _t = *ptr_y + _y;
        c = (_t - *ptr_y) - _y;
        *ptr_y = _t;
    }
}

gatherStatus_t run_scatter_grad(cudaStream_t stream, const float *grad_sum, float *grad_x, float *grad_y,
                                const unsigned int *lx, unsigned int *ly,
                                unsigned int *sumPref, unsigned int *xCumSum, unsigned int *yCumSum,
                                unsigned int V, unsigned int lx_max, unsigned int ly_max, unsigned int N)
{
    dim3 threads1(W, H);
    dim3 blocks1((lx_max + W - 1) / W, (V + H - 1) / H, N);

    kernel_fill_grad_x<<<blocks1, threads1, 0, stream>>>(grad_sum, ly, grad_x, sumPref, xCumSum, V);
    if (cudaGetLastError() != cudaSuccess)
        return GATHER_STATUS_FAILED;

    dim3 threads2(W, H);
    dim3 blocks2((ly_max + W - 1) / W, (V + H - 1) / H, N);
    kernel_fill_grad_y<<<blocks2, threads2, 0, stream>>>(grad_sum, lx, ly, grad_y, sumPref, yCumSum, V);
    if (cudaGetLastError() != cudaSuccess)
        return GATHER_STATUS_FAILED;

    return GATHER_STATUS_SUCCESS;
}

__global__ void kernel_cat(const float *x_padded, const unsigned int *lx,
                           float *x_gather, const unsigned int *memPref,
                           const unsigned int N_stride, const unsigned int T_stride,
                           unsigned int V)
{
    unsigned int t = blockIdx.x * W + threadIdx.x;
    unsigned int n = blockIdx.z;
    if (t >= lx[n])
        return;

    unsigned int v = blockIdx.y * H + threadIdx.y;
    if (v >= V)
        return;

    x_gather[memPref[n] + t * V + v] = x_padded[n * N_stride + t * T_stride + v];
}

gatherStatus_t run_gather_cat(cudaStream_t stream, const float *x_padded, const unsigned int *lx,
                              float *x_gather, const unsigned int *memPref,
                              const unsigned int N_stride, const unsigned int T_stride,
                              unsigned int N, unsigned int T, unsigned int V)
{

    dim3 threads(W, H);
    dim3 blocks((T + W - 1) / W, (V + H - 1) / H, N);

    kernel_cat<<<blocks, threads, 0, stream>>>(x_padded, lx, x_gather, memPref, N_stride, T_stride, V);
    if (cudaGetLastError() != cudaSuccess)
        return GATHER_STATUS_FAILED;

    return GATHER_STATUS_SUCCESS;
}

__global__ void kernel_pad(const float *grad_gather, const unsigned int *lx,
                           float *grad_padded, const unsigned int *memPref,
                           const unsigned int N_stride, const unsigned int T_stride,
                           unsigned int V)
{
    unsigned int t = blockIdx.x * W + threadIdx.x;
    unsigned int n = blockIdx.z;
    if (t >= lx[n])
        return;

    unsigned int v = blockIdx.y * H + threadIdx.y;
    if (v >= V)
        return;

    grad_padded[n * N_stride + t * T_stride + v] = grad_gather[memPref[n] + t * V + v];
}

gatherStatus_t run_pad_grad(cudaStream_t stream, const float *grad_gather, const unsigned int *lx,
                            float *grad_padded, const unsigned int *memPref,
                            const unsigned int N_stride, const unsigned int T_stride,
                            unsigned int N, unsigned int T, unsigned int V)
{

    dim3 threads(W, H);
    dim3 blocks((T + W - 1) / W, (V + H - 1) / H, N);

    kernel_pad<<<blocks, threads, 0, stream>>>(grad_gather, lx, grad_padded, memPref, N_stride, T_stride, V);
    if (cudaGetLastError() != cudaSuccess)
        return GATHER_STATUS_FAILED;

    return GATHER_STATUS_SUCCESS;
}