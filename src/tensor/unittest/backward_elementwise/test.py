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

    # Test 8
    a8 = torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float32, requires_grad=True)
    b8 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True)
    res8 = a8 - b8
    res8.backward(torch.ones_like(res8))
    results.append(a8.grad.numpy())
    results.append(b8.grad.numpy())

    # Test 9
    a9 = torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float32, requires_grad=True)
    b9 = torch.tensor([[2.0, 4.0], [5.0, 8.0]], dtype=torch.float32, requires_grad=True)
    res9 = a9 / b9
    res9.backward(torch.ones_like(res9))
    results.append(a9.grad.numpy())
    results.append(b9.grad.numpy())

    # Test 10
    a10 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True)
    res10 = torch.pow(a10, 3.0)
    res10.backward(torch.ones_like(res10))
    results.append(a10.grad.numpy())

    # Test 11
    a11 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True)
    b11 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32, requires_grad=True)
    c11 = torch.tensor([2.0], dtype=torch.float32, requires_grad=True)
    res11 = (a11 + b11) / (a11 * c11) - torch.pow(b11, 2.0)
    res11.backward(torch.ones_like(res11))
    results.append(a11.grad.numpy())
    results.append(b11.grad.numpy())
    results.append(c11.grad.numpy())

    # Test 12
    a12 = torch.tensor([[0.5, 1.0], [1.5, 2.0]], dtype=torch.float32, requires_grad=True)
    b12 = torch.tensor([1.0, 2.0], dtype=torch.float32, requires_grad=True)
    res12 = torch.tanh(a12 / b12) + (a12 - b12) * a12
    res12.backward(torch.ones_like(res12))
    results.append(a12.grad.numpy())
    results.append(b12.grad.numpy())

    # Test 13
    a13 = torch.tensor([[0.5, 1.0], [1.5, 2.0]], dtype=torch.float32, requires_grad=True)
    res13 = torch.exp(a13)
    res13.backward(torch.ones_like(res13))
    results.append(a13.grad.numpy())

    # Test 14
    a14 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True)
    res14 = torch.log(a14)
    res14.backward(torch.ones_like(res14))
    results.append(a14.grad.numpy())

    # Test 15
    a15 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True)
    b15 = torch.tensor([1.0, 2.0], dtype=torch.float32, requires_grad=True)
    res15 = torch.exp(a15 / b15) + torch.log(a15 * b15)
    res15.backward(torch.ones_like(res15))
    results.append(a15.grad.numpy())
    results.append(b15.grad.numpy())

    return results

if __name__ == "__main__":
    from torch_utils import dump_torch_tensor
    for res in run():
        dump_torch_tensor(torch.from_numpy(res))
