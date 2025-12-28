import torch

def run():
    results = []

    # Test 1
    a1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True)
    b1 = torch.tensor([1.0, 2.0], dtype=torch.float32, requires_grad=True)
    c1 = a1 + b1
    c1.backward(torch.ones_like(c1))
    results.append(a1.grad.numpy())
    results.append(b1.grad.numpy())

    # Test 2
    a2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True)
    b2 = torch.tensor([10.0, 20.0], dtype=torch.float32, requires_grad=True)
    c2 = a2 * b2
    c2.backward(torch.ones_like(c2))
    results.append(a2.grad.numpy())
    results.append(b2.grad.numpy())

    # Test 3
    a3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True)
    b3 = torch.tensor([1.0, 2.0], dtype=torch.float32, requires_grad=True)
    c3 = torch.tensor([[5.0, 5.0], [5.0, 5.0]], dtype=torch.float32, requires_grad=True)
    d3 = a3 * b3 + c3
    d3.backward(torch.ones_like(d3))
    results.append(a3.grad.numpy())
    results.append(b3.grad.numpy())
    results.append(c3.grad.numpy())

    # Test 4
    a4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True)
    b4 = torch.tensor([0.5, 0.5], dtype=torch.float32, requires_grad=True)
    c4 = torch.tensor([2.0], dtype=torch.float32, requires_grad=True)
    res4 = (a4 * b4) + (a4 * c4)
    res4.backward(torch.ones_like(res4))
    results.append(a4.grad.numpy())
    results.append(b4.grad.numpy())
    results.append(c4.grad.numpy())

    # Test 5
    a5 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True)
    b5 = torch.tensor([1.0, 2.0], dtype=torch.float32, requires_grad=True)
    c5 = torch.tensor([10.0], dtype=torch.float32, requires_grad=True)
    res5 = (a5 + b5) * (a5 + c5) + b5
    res5.backward(torch.ones_like(res5))
    results.append(a5.grad.numpy())
    results.append(b5.grad.numpy())
    results.append(c5.grad.numpy())

    # Test 6
    a6 = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32, requires_grad=True)
    b6 = torch.tensor([[10.0], [20.0], [30.0]], dtype=torch.float32, requires_grad=True)
    c6 = torch.tensor([[5.0]], dtype=torch.float32, requires_grad=True)
    res6 = (a6 + b6) * (a6 + c6) * (b6 + c6)
    res6.backward(torch.ones_like(res6))
    results.append(a6.grad.numpy())
    results.append(b6.grad.numpy())
    results.append(c6.grad.numpy())

    # Test 7
    a7 = torch.tensor([[-1.0, 0.0], [1.0, 2.0]], dtype=torch.float32, requires_grad=True)
    b7 = torch.tanh(a7)
    b7.backward(torch.ones_like(b7))
    results.append(a7.grad.numpy())

    return results

if __name__ == "__main__":
    from torch_utils import dump_torch_tensor
    for res in run():
        dump_torch_tensor(torch.from_numpy(res))
