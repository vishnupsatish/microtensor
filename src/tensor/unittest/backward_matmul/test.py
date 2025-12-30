import torch
from torch_utils import dump_torch_tensor

def run():
    results = []

    # Test 1
    a1 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, requires_grad=True)
    b1 = torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.float32, requires_grad=True)
    c1 = torch.matmul(a1, b1)
    c1.backward(torch.ones_like(c1))
    results.append(a1.grad.numpy())
    results.append(b1.grad.numpy())

    # Test 2
    a2 = torch.ones((2, 2, 2), dtype=torch.float32, requires_grad=True)
    b2 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, requires_grad=True)
    c2 = torch.matmul(a2, b2)
    c2.backward(torch.ones_like(c2))
    results.append(a2.grad.numpy())
    results.append(b2.grad.numpy())

    # Test 3
    a3 = torch.tensor([1, 2, 3], dtype=torch.float32, requires_grad=True)
    b3 = torch.tensor([4, 5, 6], dtype=torch.float32, requires_grad=True)
    c3 = torch.matmul(a3, b3)
    c3.backward()
    results.append(a3.grad.numpy())
    results.append(b3.grad.numpy())

    # Test 4
    x4 = torch.tensor([[0.5, -0.5]], dtype=torch.float32, requires_grad=True)
    w4 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, requires_grad=True)
    b4 = torch.tensor([0.1, 0.2], dtype=torch.float32, requires_grad=True)
    y4 = torch.tanh(torch.matmul(x4, w4) + b4)
    y4.backward(torch.ones_like(y4))
    results.append(x4.grad.numpy())
    results.append(w4.grad.numpy())
    results.append(b4.grad.numpy())

    # Test 5
    x5 = torch.ones((2, 3), dtype=torch.float32, requires_grad=True)
    w5_1 = torch.full((3, 2), 2.0, dtype=torch.float32, requires_grad=True)
    w5_2 = torch.full((3, 2), 3.0, dtype=torch.float32, requires_grad=True)
    y5 = torch.matmul(x5, w5_1) + torch.matmul(x5, w5_2)
    y5.backward(torch.ones_like(y5))
    results.append(x5.grad.numpy())
    results.append(w5_1.grad.numpy())
    results.append(w5_2.grad.numpy())

    # Test 6
    a6 = torch.ones((2, 1, 2, 2), dtype=torch.float32, requires_grad=True)
    b6 = torch.full((1, 2, 2, 2), 2.0, dtype=torch.float32, requires_grad=True)
    c6 = torch.matmul(a6, b6)
    c6.backward(torch.ones_like(c6))
    results.append(a6.grad.numpy())
    results.append(b6.grad.numpy())

    # Test 7
    x7 = torch.ones((2, 3, 2), dtype=torch.float32, requires_grad=True)
    w7 = torch.full((2, 2), 2.0, dtype=torch.float32, requires_grad=True)
    x7_p = x7.permute(1, 0, 2)
    y7 = torch.matmul(x7_p, w7)
    y7.backward(torch.ones_like(y7))
    results.append(x7.grad.numpy())
    results.append(w7.grad.numpy())

    return results

if __name__ == "__main__":
    for res in run():
        dump_torch_tensor(torch.from_numpy(res))

