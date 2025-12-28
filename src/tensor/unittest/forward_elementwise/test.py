import torch

from torch_utils import dump_torch_tensor

def run():
    results = []

    # Test 1
    a1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    b1 = torch.tensor([1.0, 2.0], dtype=torch.float32)
    results.append((a1 + b1).detach().numpy())

    # Test 2
    a2 = torch.ones((2, 3, 4), dtype=torch.float32)
    b2 = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
    results.append((a2 + b2).detach().numpy())

    # Test 3
    a3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    b3 = torch.tensor([5.0], dtype=torch.float32)
    results.append((a3 * b3).detach().numpy())

    # Test 4
    a4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    b4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    c4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    results.append((a4 + b4 * c4).detach().numpy())

    # Test 5
    a5 = torch.tensor([[-1.0, 0.0], [1.0, 2.0]], dtype=torch.float32)
    results.append(torch.tanh(a5).detach().numpy())

    return results

if __name__ == "__main__":
    # If run directly, still print for debugging
    for res in run():
        dump_torch_tensor(torch.from_numpy(res))
