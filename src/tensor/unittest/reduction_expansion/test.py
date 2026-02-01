import torch
import torch.nn.functional as F
from torch_utils import dump_torch_tensor

def run():
    results = []

    # Test 1
    a1 = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
    results.append(torch.sum(a1).numpy())

    # Test 2
    a2 = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
    results.append(torch.sum(a2, dim=0, keepdim=False).numpy())

    # Test 3
    a3 = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
    results.append(torch.sum(a3, dim=1, keepdim=True).numpy())

    # Test 4
    a4 = torch.tensor(5.0, dtype=torch.float32)
    results.append(torch.broadcast_to(a4, (2, 3)).numpy())

    # Test 5
    a5 = torch.tensor([[1, 2, 3]], dtype=torch.float32)
    results.append(torch.broadcast_to(a5, (2, 3)).numpy())

    # Test 6
    a6 = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32, requires_grad=True)
    b6 = torch.sum(a6)
    b6.backward()
    results.append(a6.grad.numpy())

    # Test 7
    a7 = torch.tensor(5.0, dtype=torch.float32, requires_grad=True)
    b7 = torch.broadcast_to(a7, (2, 3))
    b7.backward(torch.ones_like(b7))
    results.append(a7.grad.numpy())

    # Test 8
    a8 = torch.tensor([[1], [2]], dtype=torch.float32, requires_grad=True)
    b8 = torch.broadcast_to(a8, (2, 5))
    c8 = torch.sum(b8, dim=1)
    c8.backward(torch.ones_like(c8))
    results.append(a8.grad.numpy())

    # Test 9
    a9 = torch.ones((2, 3, 4), dtype=torch.float32)
    b9 = torch.sum(a9, dim=2, keepdim=True)
    c9 = torch.sum(b9, dim=0, keepdim=False)
    results.append(torch.sum(c9, dim=0, keepdim=False).numpy())

    # Test 10
    a10 = torch.ones((2, 3, 4, 2), dtype=torch.float32)
    results.append(torch.sum(a10, dim=(0, 2), keepdim=False).numpy())

    # Test 11
    a11 = torch.tensor([[1], [2], [3]], dtype=torch.float32)
    b11 = torch.broadcast_to(a11, (3, 4))
    results.append(torch.sum(b11, dim=0, keepdim=False).numpy())

    # Test 12
    a12 = torch.ones((2, 2, 2), dtype=torch.float32, requires_grad=True)
    b12 = torch.sum(a12, dim=(0, 2), keepdim=False)
    c12 = torch.sum(b12, dim=0, keepdim=False)
    c12.backward()
    results.append(a12.grad.numpy())

    # Test 13
    a13 = torch.tensor([[1, 6], [3, 4], [5, 2]], dtype=torch.float32)
    results.append(torch.max(a13, dim=0, keepdim=False).values.numpy())

    # Test 14
    a14 = torch.tensor([[1, 6], [3, 4], [5, 2]], dtype=torch.float32)
    results.append(torch.max(a14, dim=1, keepdim=True).values.numpy())

    # Test 15
    a15 = torch.tensor([[1, 6], [3, 4], [5, 2]], dtype=torch.float32, requires_grad=True)
    b15 = torch.max(a15, dim=0, keepdim=False).values
    b15.backward(torch.ones_like(b15))
    results.append(a15.grad.numpy())

    # Test 16
    a16 = torch.tensor([[1, 2], [7, 4], [5, 8]], dtype=torch.float32, requires_grad=True)
    b16 = torch.max(a16, dim=0, keepdim=False).values
    b16.backward(torch.ones_like(b16))
    results.append(a16.grad.numpy())

    # Test 17
    a17 = torch.tensor([[5, 2], [5, 4], [1, 4]], dtype=torch.float32)
    results.append(torch.max(a17, dim=0, keepdim=False).values.numpy())

    # Test 18
    a18 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, requires_grad=True)
    b18 = torch.tensor([[10, 20], [30, 40]], dtype=torch.float32, requires_grad=True)
    c18 = torch.tensor([0.5, 0.5], dtype=torch.float32, requires_grad=True)
    res18 = torch.max(a18 + b18, dim=0, keepdim=False).values * c18
    res18.backward(torch.ones_like(res18))
    results.append(a18.grad.numpy())
    results.append(b18.grad.numpy())
    results.append(c18.grad.numpy())

    # Test 19
    a19 = torch.tensor([[1, 5], [6, 3], [2, 4]], dtype=torch.float32, requires_grad=True)
    b19 = torch.tensor([1, 1, 1], dtype=torch.float32, requires_grad=True)
    res19 = torch.pow(torch.max(a19, dim=1, keepdim=False).values - b19, 2.0)
    res19.backward(torch.ones_like(res19))
    results.append(a19.grad.numpy())
    results.append(b19.grad.numpy())

    # Test 20
    a20 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    b20 = F.softmax(a20, dim=0)
    results.append(b20.detach().numpy())

    # Test 21
    a21 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    b21 = F.softmax(a21, dim=0)
    results.append(b21.detach().numpy())

    # Test 22
    a22 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    b22 = F.softmax(a22, dim=1)
    results.append(b22.detach().numpy())

    # Test 23
    a23 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, requires_grad=True)
    b23 = F.softmax(a23, dim=0)
    c23 = b23 * torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    d23 = c23.sum()
    d23.backward()
    results.append(a23.grad.numpy())

    # Test 24
    a24 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32, requires_grad=True)
    b24 = F.softmax(a24, dim=1)
    c24 = b24.sum()
    c24.backward()
    results.append(a24.grad.numpy())

    # Test 25
    a25 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True)
    b25 = torch.tensor([[0.5, 1.5], [2.5, 3.5]], dtype=torch.float32, requires_grad=True)
    res25 = F.log_softmax(a25 + b25, dim=1).sum()
    res25.backward()
    results.append(a25.grad.numpy())
    results.append(b25.grad.numpy())

    # Test 26
    a26 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
    results.append(torch.triu(a26, diagonal=0).numpy())
    results.append(torch.triu(a26, diagonal=1).numpy())
    results.append(torch.triu(a26, diagonal=-1).numpy())

    # Test 27
    a27 = torch.ones((2, 3, 3), dtype=torch.float32, requires_grad=True)
    b27 = torch.triu(a27, diagonal=1)
    c27 = b27.sum()
    c27.backward()
    results.append(a27.grad.numpy())

    # Test 28
    a28 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, requires_grad=True)
    mask28 = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)
    b28 = torch.masked_fill(a28, mask28 == 1, -1e9)
    results.append(b28.detach().numpy())
    b28.sum().backward()
    results.append(a28.grad.numpy())

    # Test 29
    a29 = torch.ones((3, 3), dtype=torch.float32, requires_grad=True)
    mask29 = torch.triu(torch.ones((3, 3)), diagonal=1)
    b29 = torch.masked_fill(a29, mask29 == 1, -1e9)
    results.append(b29.detach().numpy())
    b29.sum().backward()
    results.append(a29.grad.numpy())

    # Test 30
    a30 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    idx30 = torch.tensor([[0, 2], [1, 1]], dtype=torch.int64)
    results.append(torch.gather(a30, 1, idx30).numpy())

    # Test 31
    a31 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    idx31 = torch.tensor([[1, 0, 1]], dtype=torch.int64)
    results.append(torch.gather(a31, 0, idx31).numpy())

    # Test 32
    a32 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, requires_grad=True)
    idx32 = torch.tensor([[0, 2], [1, 1]], dtype=torch.int64)
    b32 = torch.gather(a32, 1, idx32)
    b32.sum().backward()
    results.append(a32.grad.numpy())

    # Test 33
    a33 = torch.tensor([[1, 2, 3]], dtype=torch.float32, requires_grad=True)
    idx33 = torch.tensor([[0, 0, 0, 0]], dtype=torch.int64)
    b33 = torch.gather(a33, 1, idx33)
    b33.sum().backward()
    results.append(a33.grad.numpy())

    # Test 34
    a34 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, requires_grad=True)
    idx34 = torch.tensor([[1], [0]], dtype=torch.int64)
    b34 = torch.gather(F.softmax(a34, dim=1), 1, idx34)
    b34.sum().backward()
    results.append(a34.grad.numpy())

    # Test 35
    a35 = torch.tensor([[1, 6], [3, 4], [5, 2]], dtype=torch.float32)
    results.append(torch.argmax(a35, dim=0, keepdim=False).to(torch.float32).numpy())

    # Test 36
    a36 = torch.tensor([[1, 6], [3, 4], [5, 2]], dtype=torch.float32)
    results.append(torch.argmax(a36, dim=1, keepdim=True).to(torch.float32).numpy())

    # Test 37
    a37 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    weight37 = torch.tensor([1, 1, 1], dtype=torch.float32)
    bias37 = torch.tensor([0, 0, 0], dtype=torch.float32)
    results.append(F.layer_norm(a37, (3,), weight37, bias37).numpy())

    # Test 38
    a38 = torch.tensor([[[1, 2, 3], [4, 5, 6]],
                        [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32)
    weight38 = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.float32)
    bias38 = torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.float32)
    results.append(F.layer_norm(a38, (2, 3), weight38, bias38).numpy())

    # Test 39
    a39 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32)
    weight39 = torch.tensor([2, 2, 2, 2], dtype=torch.float32)
    bias39 = torch.tensor([1, 1, 1, 1], dtype=torch.float32)
    results.append(F.layer_norm(a39, (4,), weight39, bias39).numpy())

    # Test 40
    a40 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, requires_grad=True)
    weight40 = torch.tensor([1, 1, 1], dtype=torch.float32)
    bias40 = torch.tensor([0, 0, 0], dtype=torch.float32)
    out40 = F.layer_norm(a40, (3,), weight40, bias40)
    out40.sum().backward()
    results.append(a40.grad.numpy())

    # Test 41
    a41 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    weight41 = torch.tensor([1, 1, 1], dtype=torch.float32, requires_grad=True)
    bias41 = torch.tensor([0, 0, 0], dtype=torch.float32, requires_grad=True)
    out41 = F.layer_norm(a41, (3,), weight41, bias41)
    out41.sum().backward()
    results.append(weight41.grad.numpy())
    results.append(bias41.grad.numpy())

    # Test 42
    a42 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32, requires_grad=True)
    weight42 = torch.tensor([2, 1, 0.5, 0.25], dtype=torch.float32, requires_grad=True)
    bias42 = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32, requires_grad=True)
    out42 = F.layer_norm(a42, (4,), weight42, bias42)
    loss42 = (out42 * out42).sum()
    loss42.backward()
    results.append(a42.grad.numpy())
    results.append(weight42.grad.numpy())
    results.append(bias42.grad.numpy())

    # Test 43
    a43 = torch.tensor([[[1, 2, 3], [4, 5, 6]],
                        [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32, requires_grad=True)
    weight43 = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.float32, requires_grad=True)
    bias43 = torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.float32, requires_grad=True)
    out43 = F.layer_norm(a43, (2, 3), weight43, bias43)
    out43.sum().backward()
    results.append(a43.grad.numpy())
    results.append(weight43.grad.numpy())
    results.append(bias43.grad.numpy())

    return results

if __name__ == "__main__":
    for res in run():
        dump_torch_tensor(torch.from_numpy(res))

