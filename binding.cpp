#include <tuple>
#include <string>

#include <c10/cuda/CUDAStream.h>
#include <torch/types.h>
#include <torch/extension.h>

#include "core.h"

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_CHECK
#endif

#define CHECK_CONTIGUOUS(x)          \
    TORCH_CHECK((x).is_contiguous(), \
                #x " must be contiguous")

#define CHECK_CUDA(x)                   \
    TORCH_CHECK((x).device().is_cuda(), \
                #x " must be located in the CUDA")

#define CHECK_INT(x)                                      \
    TORCH_CHECK((x).scalar_type() == at::ScalarType::Int, \
                #x " must be a Int tensor")

#define None torch::indexing::None
#define Slice torch::indexing::Slice

torch::Tensor gather_cat_forward(
    const torch::Tensor &x_padded, const torch::Tensor &lx)
{
    // CHECK_CONTIGUOUS(x_padded);      // contiguous is no longer required
    CHECK_CONTIGUOUS(lx);
    // Check types
    CHECK_INT(lx);
    // Check device
    CHECK_CUDA(x_padded);
    CHECK_CUDA(lx);
    // Check number of dimensions and elements
    TORCH_CHECK(x_padded.dim() == 3, "x_padded must have 3 dimensions (N, T, V)")
    TORCH_CHECK(x_padded.stride(2) == 1, "x_padded must has stride=1 in last dim")
    TORCH_CHECK(x_padded.size(0) == lx.size(0), "lx and x_padded in dim 0 must be equal to N")

    const auto N = x_padded.size(0);
    const auto T = x_padded.size(1);
    const auto device = x_padded.device();
    auto V = x_padded.size(2);

    auto memPref = lx.cumsum(0, at::ScalarType::Int);

    int64_t sumT = memPref[-1].item<int64_t>();

    // initialize at cuda kernel
    torch::Tensor x_gather = torch::empty({sumT, V}, torch::dtype(x_padded.scalar_type()).device(device));

    /* aligned to 16 bits */
    auto N_stride = x_padded.stride(0);
    auto T_stride = x_padded.stride(1);
    switch (x_padded.scalar_type())
    {
    case torch::kInt32:
    case torch::kFloat32:
        V *= 2;
        N_stride *= 2;
        T_stride *= 2;
        break;
    case torch::kInt64:
    case torch::kFloat64:
        V *= 4;
        N_stride *= 4;
        T_stride *= 4;
        break;
    default:
        break;
    }
    // set begin of memory location of each sequence
    {
        auto cumsumMemPref = memPref.index({Slice(0, -1, None)}) * V;
        memPref.index_put_({Slice(1, None, None)}, cumsumMemPref);
    }
    memPref[0] = 0;

    auto stream = c10::cuda::getCurrentCUDAStream(device.index());
    gatherStatus_t status;

    status = run_gather_cat(stream, (const ushort *)x_padded.data_ptr(), (unsigned int *)lx.data_ptr<int>(),
                            (ushort *)x_gather.data_ptr(), (unsigned int *)memPref.data_ptr<int>(),
                            N_stride, T_stride, N, T, V);

    TORCH_CHECK(status == GATHER_STATUS_SUCCESS, "gather cat status " + std::to_string(status));

    return x_gather;
}

torch::Tensor gather_cat_backward(
    const torch::Tensor &grad_gather, const torch::Tensor &lx,
    long &N_stride, long &T_stride)
{
    CHECK_CONTIGUOUS(grad_gather);
    CHECK_CONTIGUOUS(lx);
    // Check types
    CHECK_INT(lx);
    // Check device
    CHECK_CUDA(grad_gather);
    CHECK_CUDA(lx);
    // Check number of dimensions and elements
    TORCH_CHECK(grad_gather.dim() == 2, "grad_gather must have 2 dimensions (NT, V)")

    const auto N = lx.size(0);
    const auto T = lx.max().item<int64_t>();
    const auto device = grad_gather.device();
    auto V = grad_gather.size(1);
    torch::Tensor grad_padded = torch::zeros({N, T, V}, torch::dtype(grad_gather.scalar_type()).device(device));

    /* aligned to 16 bits */
    switch (grad_gather.scalar_type())
    {
    case torch::kInt32:
    case torch::kFloat32:
        V *= 2;
        N_stride *= 2;
        T_stride *= 2;
        break;
    case torch::kInt64:
    case torch::kFloat64:
        V *= 4;
        N_stride *= 4;
        T_stride *= 4;
        break;
    default:
        break;
    }

    auto memPref = lx.cumsum(0, at::ScalarType::Int);
    {
        auto cumsumMemPref = memPref.index({Slice(0, -1, None)}) * V;
        memPref.index_put_({Slice(1, None, None)}, cumsumMemPref);
    }
    memPref[0] = 0;

    auto stream = c10::cuda::getCurrentCUDAStream(device.index());
    gatherStatus_t status;

    status = run_pad_grad(stream, (const ushort *)grad_gather.data_ptr(), (unsigned int *)lx.data_ptr<int>(),
                          (ushort *)grad_padded.data_ptr(), (unsigned int *)memPref.data_ptr<int>(),
                          N_stride, T_stride, N, T, V);

    TORCH_CHECK(status == GATHER_STATUS_SUCCESS, "gather cat backward status " + std::to_string(status));

    return grad_padded;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "gather_cat_forward",
        &gather_cat_forward,
        "CUDA based gather cat forward",
        pybind11::arg("x_padded"),
        pybind11::arg("lx"));

    m.def(
        "gather_cat_backward",
        &gather_cat_backward,
        "CUDA based gather cat backward",
        pybind11::arg("grad_gather"),
        pybind11::arg("lx"),
        pybind11::arg("N_stride"),
        pybind11::arg("T_stride"));
}
