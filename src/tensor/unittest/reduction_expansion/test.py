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

    return results

if __name__ == "__main__":
    for res in run():
        dump_torch_tensor(torch.from_numpy(res))

