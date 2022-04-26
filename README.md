# Torch-gather
A mini library that implements several useful functions binding to PyTorch in C++.

## What does gather do? Why do we need it?

When dealing with sequences, a common way of processing the variable lengths is padding them to the max length, which leads to quite a lot redundancies and waste on computing and memory as sequences length varies. So `gather` just removes their paddings and makes computation without waste of computation resource.

## Install

```shell
python setup.py install
```

## Docs

**Note that all the input tensors should be on cuda device.**

* `gather.cat(x_padded:torch.FloatTensor, lx:torch.IntTensor)`

    Return a concatence of given padded tensor `x_padded` according to its lengths `lx`. Support mixed precision forward/backward.

    Input:

    **x_padded (`torch.float16|32|64 / torch.int16|32|64`):** padded tensor of size `(N, L, V)` or `(N, L)`, where `L=max(lx)`. 

    **lx (`torch.int32`):** lengths of size `(N, )`.

    Return:

    **x_gather (dtype the same as `x_padded`):** the gathered tensor without paddings of size `(lx[0]+lx[1]+...+lx[N-1], V)` or `(lx[0]+lx[1]+...+lx[N-1], V)` depending on shape of `x_padded`.

    Example:

    ```python
    >>> import torch
    >>> import gather
    >>> lx = torch.randint(3, 20, (5, ), dtype=torch.int32)
    >>> x_padded = torch.randn((5, lx.max(), 64), device='cuda')
    >>> x_padded.size(), lx.size()
    (torch.Size([5, 19, 64]), torch.Size([5]))
    >>> x_gather = gather.cat(x_padded, lx)
    >>> x_gather.size()
    torch.Size([81, 64])
    # another example, with dim = 2 and dtype=torch.int64
    >>> x_padded = torch.tensor([[1, 2, 3],[1, 2, 0]], device='cuda')
    >>> lx = torch.tensor([3, 2], dtype=torch.int32)
    >>> x_padded
    tensor([[1, 2, 3],
            [1, 2, 0]], device='cuda:0')
    >>> lx
    tensor([3, 2], dtype=torch.int32)
    >>> gather.cat(x_padded, lx)
    tensor([1, 2, 3, 1, 2], device='cuda:0')
    ```

    This function is easy to implement with torch python functions like `torch.cat()`, however, `gather.cat()` is *customized* for specified tasks, and more efficient.

* `gather.sum(xs:torch.FloatTensor, ys:torch.FloatTensor, lx:torch.IntTensor, ly:torch.IntTensor)`

    Return a sequence-matched broadcast sum of given paired **gathered** tensor `xs` and `ys`. For a pair of sequences in `xs` and `ys`, say `xs_i` and `ys_i`, `gather.sum()` broadcast them so that they can be added up. The broadcast step can be understood as `(xs_i.unsqueeze(1)+ys_i.unsqueeze(2)).reshape(-1, V)` with python and torch.

    Input:

    **xs (`torch.float`):** gathered tensor of size `(ST, V)`, where `ST=sum(lx)`.

    **ys (`torch.float`):** gathered tensor of size `(SU, V)`, where `SU=sum(ly)`. 

    **lx (`torch.int`):** lengths of size `(N, )`. `lx[i]` denotes length of the $i_{th}$ sequence in `xs`.

    **ly (`torch.int`):** lengths of size `(N, )`. `ly[i]` denotes length of the $i_{th}$ sequence in `ys`.

    Return:

    **gathered_sum (`torch.float`):** the gathered sequence-match sum of size `(lx[0]ly[0]+lx[1]ly[1]+...+lx[N-1]ly[N-1], V)`

    Example:

    ```python
    >>> import torch
    >>> import gather
    >>> N, T, U, V = 5, 4, 4, 3
    >>> lx = torch.randint(1, T, (N, ), dtype=torch.int32, device='cuda')
    >>> ly = torch.randint(1, U, (N, ), dtype=torch.int32, device='cuda')
    >>> xs = torch.randn((lx.sum(), V), device='cuda')
    >>> ys = torch.randn((ly.sum(), V), device='cuda')
    >>> xs.size(), ys.size(), lx.size(), ly.size()
    (torch.Size([11, 3]), torch.Size([10, 3]), torch.Size([5]), torch.Size([5]))
    >>> gathered_sum = gather.sum(xs, ys, lx, ly)
    >>> gathered_sum.size()
    torch.Size([20, 3])
    # let's see how the size 20 comes out
    >>> lx.tolist(), ly.tolist()
    ([2, 2, 1, 3, 3], [3, 1, 3, 1, 2])
    # still unclear? Uh, how about this?
    >>> (lx * ly).sum().item()
    20
    ```

    This function seems doing something weird. Please refer to the discussion page for a specific usage example.


## Reference

* PyTorch binding refers to the [1ytic/warp-rnnt](https://github.com/1ytic/warp-rnnt)

* For the specific usage of these functions, please refer to [this](https://github.com/1ytic/warp-rnnt/pull/26#issuecomment-914103575) discussion.
