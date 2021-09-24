import torch
import gather._C as core
from pkg_resources import get_distribution

__version__ = get_distribution('gather').version


class _GatherSum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, xs, ys, lx, ly):
        gathered_sum = core.gather_sum_forward(xs, ys, lx, ly)
        ctx.save_for_backward(lx, ly)
        return gathered_sum

    @staticmethod
    def backward(ctx, grad_sum):

        lx, ly = ctx.saved_tensors
        grad_x, grad_y = core.gather_sum_backward(
            grad_sum.contiguous(), lx, ly)
        return grad_x, grad_y, None, None


class _GatherCat(torch.autograd.Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, xs: torch.FloatTensor, lx: torch.IntTensor):
        x_gather = core.gather_cat_forward(xs, lx)
        ctx.nstride, ctx.tstride = xs.stride()[:-1]
        ctx.save_for_backward(lx)
        return x_gather

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_gather):

        lx, = ctx.saved_tensors
        grad_padded = core.gather_cat_backward(
            grad_gather.contiguous(), lx, ctx.nstride, ctx.tstride)
        return grad_padded, None


def sum(xs: torch.Tensor, ys: torch.Tensor, lx: torch.Tensor, ly: torch.Tensor) -> torch.Tensor:
    """ Sum the two 'gathered' tensors xs and ys.

    Args:
        xs (torch.FloatTensor): of size (lx0+lx1+..., *)
        ys (torch.FloatTensor): of size (ly0+ly1+..., *)
        lx (torch.LongTensor): of size (N, )
        ly (torch.LongTensor): of size (N, )

    Return:
        gathered_sum (torch.FloatTensor): size (lx0ly0+lx1ly1+..., *)
    """
    return _GatherSum.apply(xs, ys, lx.to(device=xs.device, dtype=torch.int32), ly.to(device=xs.device, dtype=torch.int32))


def cat(xs: torch.Tensor, lx: torch.Tensor) -> torch.Tensor:
    """Cat the padded xs via lengths lx

    Args:
        xs (torch.FloatTensor): of size (N, T, V)
        lx (torch.LongTensor): of size (N, ), whose elements are (lx0, lx1, ...)

    Return:
        x_gather (torch.FloatTensor): size (lx0+lx1+..., V)
    """
    assert xs.size(0) == lx.size(0)
    return _GatherCat.apply(xs, lx.to(device=xs.device, dtype=torch.int32))
