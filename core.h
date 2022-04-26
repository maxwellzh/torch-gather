#ifndef GATHER_CORE_H
#define GATHER_CORE_H

#include <cuda_runtime.h>

typedef enum
{
    GATHER_STATUS_SUCCESS = 0,
    GATHER_STATUS_FAILED = 1
} gatherStatus_t;

#ifdef __cplusplus
#include <cstddef>
extern "C"
{
#endif

    gatherStatus_t run_gather_cat(
        cudaStream_t stream, const ushort *x_padded, const unsigned int *lx,
        ushort *x_gather, const unsigned int *memPref,
        const unsigned int N_stride, const unsigned int T_stride,
        unsigned int N, unsigned int T, unsigned int V);

    gatherStatus_t run_pad_grad(
        cudaStream_t stream, const ushort *grad_gather, const unsigned int *lx,
        ushort *grad_padded, const unsigned int *memPref,
        const unsigned int N_stride, const unsigned int T_stride,
        unsigned int N, unsigned int T, unsigned int V);

#ifdef __cplusplus
}
#endif

#endif
