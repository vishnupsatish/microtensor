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

    # Test 17: Forward reduceMax with tied maximum values
    a17 = torch.tensor([[5, 2], [5, 4], [1, 4]], dtype=torch.float32)
    results.append(torch.max(a17, dim=0, keepdim=False).values.numpy())

    # Test 18: (a + b).reduceMax(0) * c
    a18 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, requires_grad=True)
    b18 = torch.tensor([[10, 20], [30, 40]], dtype=torch.float32, requires_grad=True)
    c18 = torch.tensor([0.5, 0.5], dtype=torch.float32, requires_grad=True)
    res18 = torch.max(a18 + b18, dim=0, keepdim=False).values * c18
    res18.backward(torch.ones_like(res18))
    results.append(a18.grad.numpy())
    results.append(b18.grad.numpy())
    results.append(c18.grad.numpy())

    # Test 19: (a.reduceMax(1) - b).pow(2.0)
    a19 = torch.tensor([[1, 5], [6, 3], [2, 4]], dtype=torch.float32, requires_grad=True)
    b19 = torch.tensor([1, 1, 1], dtype=torch.float32, requires_grad=True)
    res19 = torch.pow(torch.max(a19, dim=1, keepdim=False).values - b19, 2.0)
    res19.backward(torch.ones_like(res19))
    results.append(a19.grad.numpy())
    results.append(b19.grad.numpy())

    # Test 20: Basic 1D Softmax Forward
    a20 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    b20 = F.softmax(a20, dim=0)
    results.append(b20.detach().numpy())

    # Test 21: 2D Softmax Forward along dim 0
    a21 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    b21 = F.softmax(a21, dim=0)
    results.append(b21.detach().numpy())

    # Test 22: 2D Softmax Forward along dim 1
    a22 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    b22 = F.softmax(a22, dim=1)
    results.append(b22.detach().numpy())

    # Test 23: Softmax Backward 1D
    a23 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, requires_grad=True)
    b23 = F.softmax(a23, dim=0)
    c23 = b23 * torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    d23 = c23.sum()
    d23.backward()
    results.append(a23.grad.numpy())

    # Test 24: Softmax Backward 2D dim 1
    a24 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32, requires_grad=True)
    b24 = F.softmax(a24, dim=1)
    c24 = b24.sum()
    c24.backward()
    results.append(a24.grad.numpy())

    # Test 25: Complex chain with Softmax
    a25 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True)
    b25 = torch.tensor([[0.5, 1.5], [2.5, 3.5]], dtype=torch.float32, requires_grad=True)
    res25 = F.log_softmax(a25 + b25, dim=1).sum()
    res25.backward()
    results.append(a25.grad.numpy())
    results.append(b25.grad.numpy())

    # Test 26: triu forward
    a26 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
    results.append(torch.triu(a26, diagonal=0).numpy())
    results.append(torch.triu(a26, diagonal=1).numpy())
    results.append(torch.triu(a26, diagonal=-1).numpy())

    # Test 27: triu backward
    a27 = torch.ones((2, 3, 3), dtype=torch.float32, requires_grad=True)
    b27 = torch.triu(a27, diagonal=1)
    c27 = b27.sum()
    c27.backward()
    results.append(a27.grad.numpy())

    # Test 28: maskedFill basic
    a28 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, requires_grad=True)
    mask28 = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)
    b28 = torch.masked_fill(a28, mask28 == 1, -1e9)
    results.append(b28.detach().numpy())
    b28.sum().backward()
    results.append(a28.grad.numpy())

    # Test 29: Causal mask style
    a29 = torch.ones((3, 3), dtype=torch.float32, requires_grad=True)
    mask29 = torch.triu(torch.ones((3, 3)), diagonal=1)
    b29 = torch.masked_fill(a29, mask29 == 1, -1e9)
    results.append(b29.detach().numpy())
    b29.sum().backward()
    results.append(a29.grad.numpy())

    return results

if __name__ == "__main__":
    for res in run():
        dump_torch_tensor(torch.from_numpy(res))

