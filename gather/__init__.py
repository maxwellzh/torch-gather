import torch
import gather._C as core
from pkg_resources import get_distribution

__version__ = get_distribution('gather').version


class _GatherCat(torch.autograd.Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, xs: torch.Tensor, lx: torch.IntTensor):
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

# NOTE:
# I found that for the most of cases in speech recognition, my impl CUDA binding gather.sum
# in preivous version is even slower than the torch.cat way. So let's just use torch.cat to impl gather.sum


def sum(xs: torch.Tensor, ys: torch.Tensor, lx: torch.Tensor, ly: torch.Tensor) -> torch.Tensor:
    """ Sum the two 'gathered' tensors xs and ys.

    Args:
        xs (torch.FloatTensor): of size (lx0+lx1+..., *)
        ys (torch.FloatTensor): of size (ly0+ly1+..., *)
        lx (torch.LongTensor): of size (N, )
        ly (torch.LongTensor): of size (N, )

    Return:
        gathered_sum (torch.FloatTensor): size (lx0ly0+lx1ly1+..., *)

    Examples:
    >>> # just a dummy code
    >>> xs = [1,2,3,1,1,2]
    >>> lx = [3,1,2]
    >>> ys = [1,1,2,1]
    >>> ly = [1,2,1]
    >>> gather.sum(xs, ys, lx, ly) -> [1+1, 2+1, 3+1, 1+1, 1+2, 1+1, 2+1]
    """
    assert xs.dim() == 2
    assert lx.dim() == 1
    assert xs.dim() == ys.dim()
    assert lx.dim() == ly.dim()
    assert lx.size(0) == ly.size(0)
    assert xs.size(-1) == ys.size(-1), \
        f"expect the two tensor xs, ys has same last dim, instead:\n" \
        f"xs.shape = {xs.shape} and ys.shape = {ys.shape}"

    # A more readable code:
    # out = []
    # for n in range(lx.size(0)):
    #     Ti = xs[lx_cumsun[n]-lx[n]:lx_cumsun[n], :]
    #     Ui = ys[ly_cumsun[n]-ly[n]:ly_cumsun[n], :]
    #     out.append(Ti[:, None, :] + Ui[None, :, :])
    # return torch.cat([x.view(-1, xs.size(-1)) for x in out], dim=0)

    lx_cumsun = lx.cumsum(0)
    ly_cumsun = ly.cumsum(0)
    V = xs.size(-1)
    return torch.cat([
        (
            xs[lx_cumsun[n]-lx[n]:lx_cumsun[n], :][:, None, :] +
            ys[ly_cumsun[n]-ly[n]:ly_cumsun[n], :][None, :, :]
        ).view(-1, V)
        for n in range(lx.size(0))
    ], dim=0)


def cat(xs: torch.Tensor, lx: torch.Tensor) -> torch.Tensor:
    """Cat the padded xs via lengths lx

    Args:
        xs (torch.Tensor): of size (N, T, V) or (N, T)
        lx (torch.IntTensor): of size (N, ), whose elements are (lx0, lx1, ...)

    Return:
        x_gather (torch.Tensor): size (lx0+lx1+..., V) or (lx0+lx1+..., ) depending on input dim
    """
    assert xs.dtype in \
        [torch.int16, torch.int32, torch.int64, torch.float, torch.float16,
            torch.float64], f"expect xs to be torch.int<16,32,64>/torch.float<16,32,64> type, instead of {xs.dtype}"

    assert xs.size(0) == lx.size(0)
    if xs.dim() == 3:
        return _GatherCat.apply(xs.contiguous(), lx.to(device=xs.device, dtype=torch.int32))
    elif xs.dim() == 2:
        return _GatherCat.apply(xs.contiguous().unsqueeze(2), lx.to(device=xs.device, dtype=torch.int32)).squeeze(1)
    else:
        raise ValueError(
            f"gather.cat(): input xs has {xs.dim()} dimensions, expected one of [2, 3]")


__all__ = [sum, cat]
