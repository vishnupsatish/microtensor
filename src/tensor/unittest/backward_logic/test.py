import torch
import numpy as np

def run():
    results = []

    # Test 1: requiresGrad mix 1
    a1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True)
    b1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32, requires_grad=False)
    c1 = a1 * b1
    c1.backward(torch.ones_like(c1))
    results.append(a1.grad.numpy())
    results.append(b1.grad.numpy() if b1.grad is not None else np.zeros_like(b1.detach().numpy()))

    # Test 2: requiresGrad mix 2 (chained)
    a2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=False)
    b2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32, requires_grad=True)
    c2 = (a2 + b2) * a2
    c2.backward(torch.ones_like(c2))
    results.append(a2.grad.numpy() if a2.grad is not None else np.zeros_like(a2.detach().numpy()))
    results.append(b2.grad.numpy())

    # Test 3: all requiresGrad false
    a3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=False)
    b3 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32, requires_grad=False)
    c3 = a3 + b3
    if c3.requires_grad:
        c3.backward(torch.ones_like(c3))
    results.append(a3.grad.numpy() if a3.grad is not None else np.zeros_like(a3.detach().numpy()))
    results.append(b3.grad.numpy() if b3.grad is not None else np.zeros_like(b3.detach().numpy()))

    # Test 4: (a + b) * a where (a + b) is in NoGrad
    a4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True)
    b4 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        z4 = a4 + b4
    c4 = z4 * a4
    c4.backward(torch.ones_like(c4))
    results.append(a4.grad.numpy())
    results.append(b4.grad.numpy() if b4.grad is not None else np.zeros_like(b4.detach().numpy()))

    # Test 5: a.exp() * a where a.exp() is in NoGrad
    a5 = torch.tensor([[0.5, 1.0], [1.5, 2.0]], dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        z5 = torch.exp(a5)
    c5 = z5 * a5
    c5.backward(torch.ones_like(c5))
    results.append(a5.grad.numpy())

    # Test 6: (a * b) + (a * b) where one is in NoGrad
    a6 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True)
    b6 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        z1 = a6 * b6
    z2 = a6 * b6
    res6 = z1 + z2
    res6.backward(torch.ones_like(res6))
    results.append(a6.grad.numpy())
    results.append(b6.grad.numpy())

    return results

if __name__ == "__main__":
    from torch_utils import dump_torch_tensor
    for res in run():
        dump_torch_tensor(torch.from_numpy(res))

