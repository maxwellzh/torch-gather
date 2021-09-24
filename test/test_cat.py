
from gather._C import gather_cat_forward, gather_cat_backward
from gather import cat as gathercat
import torch
import time


def vis(msg: str, L: int = 40):
    if len(msg) >= L:
        print(msg)
    else:
        pad_l = (L-len(msg))//2
        pad_r = (L-len(msg)) - pad_l
        print("{} {} {}".format(pad_l*'=', msg, pad_r*'='))


def test(seed: int):
    vis(f'Test process with seed={seed}', 60)
    torch.manual_seed(seed)

    N = torch.randint(1, 32, (1,)).item()
    T = torch.randint(2, 512, (1,)).item()
    V = torch.randint(1, 1024, (1,)).item()
    lx = torch.randint(T//2, T, (N, ), dtype=torch.int, device=0)
    xs = torch.randn((N, lx.max(), V), dtype=torch.float, device=0)

    lx = lx.to(dtype=torch.int, device=0)
    print("xs size: ", xs.size())
    print("lx size: ", lx.size())

    xs.requires_grad = True

    def manual_cat(xs, lx):
        return torch.cat([xs[i, :lx[i]].view(-1, xs.size(-1)) for i in range(lx.size(0))], dim=0)

    def test_forward():
        vis('Test forward/backward computation')

        # manually cal
        manual = manual_cat(xs, lx)

        gather_x = gathercat(xs, lx)

        if not torch.all(manual == gather_x):
            print("Forward mismatch")
            print(manual)
            print(gather_x)
            raise RuntimeError
        else:
            print("Forward correct.")

        weighted_w = torch.randn_like(gather_x)
        (gather_x*weighted_w).sum().backward()
        tx_grad = xs.grad.data.detach()
        xs.grad = None

        (manual*weighted_w).sum().backward()
        mx_grad = xs.grad.data.detach()
        xs.grad = None

        cmp = tx_grad == mx_grad
        if not torch.all(cmp):
            print("Backward mismatch.")
            print(torch.sum(torch.abs(tx_grad-mx_grad)))
            print(tx_grad[torch.logical_not(cmp)])
            print(mx_grad[torch.logical_not(cmp)])
            raise RuntimeError

        else:
            print("Backward correct.")

    def test_autogradcheck():
        vis('Test autograd with torch')
        try:
            torch.autograd.gradcheck(gathercat, (xs, lx))
        except Exception as e:
            print(e)
            print("Maybe limit the (N, T, V) to smaller number and re-test.")
            exit(1)

    def test_contiguous(xs):
        model = torch.nn.LSTM(xs.size(-1), xs.size(-1),
                              num_layers=3).to(device=0)
        model.flatten_parameters()

        with torch.no_grad():
            xs, _ = model(xs.transpose(0, 1))
            xs = xs.transpose(0, 1)
        xs.requires_grad = True
        gather_x = gathercat(xs, lx)
        print("Wow! It works with non contiguous layout!")

    def test_fp16(xs:torch.Tensor):
        # xs_half = xs.to(dtype=torch.float16)
        xs_half = xs
        with torch.cuda.amp.autocast():
            gathered_x = gathercat(xs_half, lx)
            manual_x = manual_cat(xs_half, lx)

            gathered_x.sum().backward()
            g_grad = xs.grad
            xs.grad = None

            manual_x.sum().backward()
            m_grad = xs.grad
            xs.grad = None

        if torch.all(g_grad == m_grad):
            print("FP16 backward correct.")
        else:
            print(":( FP16 backward error.")

        if torch.all(gathered_x == manual_x):
            print("Wow! It works with FP16!")
        else:
            print(gathered_x)
            print(manual_x)
        pass

    def test_performance():
        with torch.no_grad():
            gather_x = gathercat(xs, lx)
            weighted = torch.randn_like(gather_x)

        cnt = 500

        t_beg = time.time()
        for _ in range(cnt):
            gather_x = manual_cat(xs, lx)
            (weighted*gather_x).mean().backward()
            xs.grad = None
        print("Torch cat runs {} times, {:.4f} ms on average".format(
            cnt, (time.time()-t_beg)/cnt*1000))

        t_beg = time.time()
        for _ in range(cnt):
            gather_x = gathercat(xs, lx)
            (weighted*gather_x).mean().backward()
            xs.grad = None

        print("Gather cat runs {} times, {:.4f} ms on average".format(
            cnt, (time.time()-t_beg)/cnt*1000))

    test_forward()
    # test_autogradcheck()
    test_contiguous(xs)
    test_fp16(xs)
    test_performance()

    print('')


if __name__ == "__main__":

    for i in range(5):
        test(i)
