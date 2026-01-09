import torch
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

    # Test 20: reshape
    a20 = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32)
    results.append(torch.reshape(a20, (2, 3)).numpy())

    # Test 21: reshape backward
    a21 = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32, requires_grad=True)
    b21 = torch.reshape(a21, (3, 2))
    b21.backward(torch.ones_like(b21))
    results.append(a21.grad.numpy())

    # Test 22: contiguous after permute
    a22 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    b22 = a22.permute(1, 0).contiguous()
    results.append(b22.numpy())

    # Test 23: contiguous after broadcast
    a23 = torch.tensor([1, 2, 3], dtype=torch.float32)
    b23 = torch.broadcast_to(a23, (2, 3)).contiguous()
    results.append(b23.numpy())

    # Test 24: reshape on non-contiguous tensor
    a24 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, requires_grad=True)
    b24 = a24.permute(1, 0) # Shape (3, 2)
    c24 = torch.reshape(b24, (6,))
    c24.backward(torch.ones_like(c24))
    results.append(a24.grad.numpy())

    return results

if __name__ == "__main__":
    for res in run():
        dump_torch_tensor(torch.from_numpy(res))

