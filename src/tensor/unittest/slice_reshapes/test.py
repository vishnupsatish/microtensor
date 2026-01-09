import torch
from torch_utils import dump_torch_tensor

def run():
    results = []

    # Test 1: reshape
    a1 = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32)
    results.append(torch.reshape(a1, (2, 3)).numpy())

    # Test 2: reshape backward
    a2 = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32, requires_grad=True)
    b2 = torch.reshape(a2, (3, 2))
    b2.backward(torch.ones_like(b2))
    results.append(a2.grad.numpy())

    # Test 3: contiguous after permute
    a3 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    b3 = a3.permute(1, 0).contiguous()
    results.append(b3.numpy())

    # Test 4: contiguous after broadcast
    a4 = torch.tensor([1, 2, 3], dtype=torch.float32)
    b4 = torch.broadcast_to(a4, (2, 3)).contiguous()
    results.append(b4.numpy())

    # Test 5: reshape on non-contiguous tensor
    a5 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, requires_grad=True)
    b5 = a5.permute(1, 0) # Shape (3, 2)
    c5 = torch.reshape(b5, (6,))
    c5.backward(torch.ones_like(c5))
    results.append(a5.grad.numpy())

    # Test 6: slice forward
    a6 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=torch.float32)
    results.append(a6[1:3, 1:3].numpy())

    # Test 7: slice backward
    a7 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=torch.float32, requires_grad=True)
    b7 = a7[1:3, 1:3]
    b7.backward(torch.ones_like(b7))
    results.append(a7.grad.numpy())

    # Test 8: slice -> permute -> add
    a8 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=torch.float32, requires_grad=True)
    b8 = a8[1:3, 0:3]
    c8 = b8.permute(1, 0)
    d8 = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=torch.float32)
    res8 = torch.pow(c8 + d8, 2.0)
    res8.backward(torch.ones_like(res8))
    results.append(a8.grad.numpy())

    # Test 9: reshape -> slice -> backward
    a9 = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=torch.float32, requires_grad=True)
    b9 = torch.reshape(a9, (3, 4))
    c9 = b9[1:3, 1:3]
    c9.backward(torch.ones_like(c9))
    results.append(a9.grad.numpy())

    # Test 10: slice -> broadcast -> backward
    a10 = torch.ones((4, 4), dtype=torch.float32, requires_grad=True)
    b10 = a10[1:2, 1:3]
    c10 = torch.broadcast_to(b10, (3, 2))
    c10.backward(torch.ones_like(c10))
    results.append(a10.grad.numpy())

    # Test 11: slice -> matmul -> backward
    a11 = torch.ones((4, 4), requires_grad=True)
    b11 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, requires_grad=True)
    a_slice11 = a11[1:3, 1:3]
    res11 = torch.matmul(a_slice11, b11)
    res11.backward(torch.ones_like(res11))
    results.append(a11.grad.numpy())
    results.append(b11.grad.numpy())

    # Test 12: reshape -> permute -> slice -> backward
    a12 = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=torch.float32, requires_grad=True)
    b12 = torch.reshape(a12, (2, 3, 2))
    c12 = b12.permute(1, 0, 2)
    d12 = c12[1:3, 0:1, 0:2]
    d12.backward(torch.ones_like(d12))
    results.append(a12.grad.numpy())

    # Test 13: matmul -> reshape -> slice -> add -> backward
    a13 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, requires_grad=True)
    b13 = torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.float32, requires_grad=True)
    c13 = torch.matmul(a13, b13)
    d13 = torch.reshape(c13, (4,))
    e13 = d13[1:3]
    res13 = e13 + torch.tensor([0.5, 0.5], dtype=torch.float32)
    res13.backward(torch.ones_like(res13))
    results.append(a13.grad.numpy())
    results.append(b13.grad.numpy())

    return results

if __name__ == "__main__":
    for res in run():
        dump_torch_tensor(torch.from_numpy(res))

