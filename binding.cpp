#include <tuple>
#include <string>

#include <THC/THC.h>

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

#define CHECK_FLOAT(x)                                      \
    TORCH_CHECK((x).scalar_type() == at::ScalarType::Float, \
                #x " must be a Float tensor")

#define CHECK_INT(x)                                      \
    TORCH_CHECK((x).scalar_type() == at::ScalarType::Int, \
                #x " must be a Int tensor")

#define None torch::indexing::None
#define Slice torch::indexing::Slice

torch::Tensor gather_sum_forward(
    const torch::Tensor &xs, const torch::Tensor &ys,
    const torch::Tensor &lx, const torch::Tensor &ly)
{
    // Check contiguous
    CHECK_CONTIGUOUS(xs);
    CHECK_CONTIGUOUS(ys);
    CHECK_CONTIGUOUS(lx);
    CHECK_CONTIGUOUS(ly);
    // Check types
    CHECK_FLOAT(xs);
    CHECK_FLOAT(ys);
    CHECK_INT(lx);
    CHECK_INT(ly);
    // Check device
    CHECK_CUDA(xs);
    CHECK_CUDA(ys);
    CHECK_CUDA(lx);
    CHECK_CUDA(ly);
    // Check number of dimensions and elements
    TORCH_CHECK(xs.dim() == 2, "xs must have 2 dimensions")
    TORCH_CHECK(ys.dim() == 2, "ys must have 2 dimensions")
    TORCH_CHECK(lx.size(0) == ly.size(0), "lx and ly shape must be equal (N,)")
    TORCH_CHECK(xs.size(0) == lx.sum().item<int64_t>(), "xs shape must be equal to (sum(lx), )")
    TORCH_CHECK(ys.size(0) == ly.sum().item<int64_t>(), "ys shape must be equal to (sum(ly), )")

    const auto N = lx.size(0);
    const auto T = lx.max().item<int64_t>(); // max of {T_i}
    const auto U = ly.max().item<int64_t>(); // max of {U_i}
    const auto V = xs.size(1);

    auto memPref = (lx * ly).cumsum(0, at::ScalarType::Int);
    auto labelPref = ly.cumsum(0, at::ScalarType::Int);
    auto framePref = lx.cumsum(0, at::ScalarType::Int);

    int64_t STU = memPref[-1].item<int64_t>();

    // set begin of memory location of each sequence
    {
        auto cumsumMemPref = memPref.index({Slice(0, -1, None)}) * V;
        auto cumsumLablePref = labelPref.index({Slice(0, -1, None)}) * V;
        auto cumsumFramePref = framePref.index({Slice(0, -1, None)}) * V;
        memPref.index_put_({Slice(1, None, None)}, cumsumMemPref);
        labelPref.index_put_({Slice(1, None, None)}, cumsumLablePref);
        framePref.index_put_({Slice(1, None, None)}, cumsumFramePref);
    }
    memPref[0] = 0;
    labelPref[0] = 0;
    framePref[0] = 0;

    const auto device = xs.device();
    auto stream = c10::cuda::getCurrentCUDAStream(device.index());
    gatherStatus_t status;

    torch::Tensor gather_xs = torch::empty({STU, V}, torch::dtype(torch::kFloat32).device(device));

    status = run_gather_sum(stream, xs.data_ptr<float>(), ys.data_ptr<float>(),
                            (unsigned int *)lx.data_ptr<int>(), (unsigned int *)ly.data_ptr<int>(),
                            gather_xs.data_ptr<float>(), (unsigned int *)memPref.data_ptr<int>(),
                            (unsigned int *)framePref.data_ptr<int>(), (unsigned int *)labelPref.data_ptr<int>(),
                            N, T, U, V);

    TORCH_CHECK(status == GATHER_STATUS_SUCCESS, "gather sum status " + std::to_string(status));

    return gather_xs;
}

std::tuple<torch::Tensor, torch::Tensor> gather_sum_backward(
    const torch::Tensor &grad_sum,
    const torch::Tensor &lx, const torch::Tensor &ly)
{
    // Check contiguous
    CHECK_CONTIGUOUS(grad_sum);
    CHECK_CONTIGUOUS(lx);
    CHECK_CONTIGUOUS(ly);
    // Check types
    CHECK_FLOAT(grad_sum);
    CHECK_INT(lx);
    CHECK_INT(ly);
    // Check device
    CHECK_CUDA(grad_sum);
    CHECK_CUDA(lx);
    CHECK_CUDA(ly);
    // Check number of dimensions and elements
    TORCH_CHECK(grad_sum.dim() == 2, "grad_sum must have 2 dimensions")
    TORCH_CHECK(lx.size(0) == ly.size(0), "lx and ly shape must be equal (N,)")

    const auto N = lx.size(0);
    const int lx_max = lx.max().item<int>(); // max of {T_i}
    const int ly_max = ly.max().item<int>(); // max of {U_i}
    const int V = grad_sum.size(1);

    const auto device = grad_sum.device();
    auto stream = c10::cuda::getCurrentCUDAStream(device.index());

    gatherStatus_t status;

    auto sumPref = (lx * ly).cumsum(0, at::ScalarType::Int);
    auto ycumsum = ly.cumsum(0, at::ScalarType::Int);
    auto xcumsum = lx.cumsum(0, at::ScalarType::Int);
    TORCH_CHECK(sumPref[-1].item<int>() == grad_sum.size(0), "grad_sum must be equal to (lx0ly0+lx1ly1+..., V)");

    // set begin of memory location of each sequence
    {
        auto cumsumsumPref = sumPref.index({Slice(0, -1, None)}) * V;
        sumPref.index_put_({Slice(1, None, None)}, cumsumsumPref);
    }
    sumPref[0] = 0;

    /**
     * Here we use torch::empty instead of torch::zeros, since we initialize the value to 0.0f in cuda kernel
     * ... and it is faster.
     */
    torch::Tensor grad_x = torch::empty({lx.sum(0).item<int>(), V}, torch::dtype(torch::kFloat32).device(device));
    torch::Tensor grad_y = torch::empty({ly.sum(0).item<int>(), V}, torch::dtype(torch::kFloat32).device(device));

    status = run_scatter_grad(stream, grad_sum.data_ptr<float>(), grad_x.data_ptr<float>(), grad_y.data_ptr<float>(),
                              (unsigned int *)lx.data_ptr<int>(), (unsigned int *)ly.data_ptr<int>(),
                              (unsigned int *)sumPref.data_ptr<int>(), (unsigned int *)xcumsum.data_ptr<int>(), (unsigned int *)ycumsum.data_ptr<int>(),
                              V, lx_max, ly_max, N);
    TORCH_CHECK(status == GATHER_STATUS_SUCCESS, "scatter grad status " + std::to_string(status));

    return std::make_tuple(grad_x, grad_y);
}

torch::Tensor gather_cat_forward(
    const torch::Tensor &x_padded, const torch::Tensor &lx)
{
    // CHECK_CONTIGUOUS(x_padded);      // contiguous is no longer required
    CHECK_CONTIGUOUS(lx);
    // Check types
    CHECK_FLOAT(x_padded);
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
    const auto V = x_padded.size(2);

    auto memPref = lx.cumsum(0, at::ScalarType::Int);

    int64_t NT = memPref[-1].item<int64_t>();

    // set begin of memory location of each sequence
    {
        auto cumsumMemPref = memPref.index({Slice(0, -1, None)}) * V;
        memPref.index_put_({Slice(1, None, None)}, cumsumMemPref);
    }
    memPref[0] = 0;

    const auto device = x_padded.device();
    auto stream = c10::cuda::getCurrentCUDAStream(device.index());
    gatherStatus_t status;

    // initialize at cuda kernel
    torch::Tensor x_gather = torch::empty({NT, V}, torch::dtype(torch::kFloat32).device(device));

    status = run_gather_cat(stream, x_padded.data_ptr<float>(), (unsigned int *)lx.data_ptr<int>(),
                            x_gather.data_ptr<float>(), (unsigned int *)memPref.data_ptr<int>(),
                            x_padded.stride(0), x_padded.stride(1), N, T, V);

    TORCH_CHECK(status == GATHER_STATUS_SUCCESS, "gather cat status " + std::to_string(status));

    return x_gather;
}

torch::Tensor gather_cat_backward(
    const torch::Tensor &grad_gather, const torch::Tensor &lx,
    const long &N_stride, const long &T_stride)
{
    CHECK_CONTIGUOUS(grad_gather);
    CHECK_CONTIGUOUS(lx);
    // Check types
    CHECK_FLOAT(grad_gather);
    CHECK_INT(lx);
    // Check device
    CHECK_CUDA(grad_gather);
    CHECK_CUDA(lx);
    // Check number of dimensions and elements
    TORCH_CHECK(grad_gather.dim() == 2, "grad_gather must have 2 dimensions (NT, V)")

    const auto N = lx.size(0);
    const auto T = lx.max().item<int64_t>();
    const auto V = grad_gather.size(1);

    auto memPref = lx.cumsum(0, at::ScalarType::Int);
    {
        auto cumsumMemPref = memPref.index({Slice(0, -1, None)}) * V;
        memPref.index_put_({Slice(1, None, None)}, cumsumMemPref);
    }
    memPref[0] = 0;

    const auto device = grad_gather.device();
    auto stream = c10::cuda::getCurrentCUDAStream(device.index());
    gatherStatus_t status;

    torch::Tensor grad_padded = torch::zeros({N, T, V}, torch::dtype(torch::kFloat32).device(device));

    status = run_pad_grad(stream, grad_gather.data_ptr<float>(), (unsigned int *)lx.data_ptr<int>(),
                          grad_padded.data_ptr<float>(), (unsigned int *)memPref.data_ptr<int>(),
                          N_stride, T_stride, N, T, V);

    TORCH_CHECK(status == GATHER_STATUS_SUCCESS, "gather cat backward status " + std::to_string(status));

    return grad_padded;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "gather_sum_forward",
        &gather_sum_forward,
        "CUDA based gather sum.",
        pybind11::arg("xs"),
        pybind11::arg("ys"),
        pybind11::arg("lx"),
        pybind11::arg("ly"));

    m.def(
        "gather_sum_backward",
        &gather_sum_backward,
        "CUDA based gather sum backward",
        pybind11::arg("grad_sum"),
        pybind11::arg("lx"),
        pybind11::arg("ly"));

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
