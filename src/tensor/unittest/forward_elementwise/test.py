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

    # Test 6
    a6 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    b6 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
    results.append((a6 - b6).detach().numpy())

    # Test 7
    a7 = torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float32)
    b7 = torch.tensor([[2.0, 4.0], [5.0, 8.0]], dtype=torch.float32)
    results.append((a7 / b7).detach().numpy())

    # Test 8
    a8 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    results.append(torch.pow(a8, 2.0).detach().numpy())

    # Test 9
    a9 = torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float32)
    b9 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    results.append(((a9 + b9) / (a9 - b9)).detach().numpy())

    # Test 10
    a10 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    b10 = torch.tensor([0.5], dtype=torch.float32)
    results.append((torch.pow(a10 * b10, 2.0) + torch.tanh(a10 - b10)).detach().numpy())

    # Test 11
    a11 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    results.append(torch.exp(a11).detach().numpy())

    # Test 12
    a12 = torch.tensor([[1.0, 2.7182818], [7.389056, 20.085537]], dtype=torch.float32)
    results.append(torch.log(a12).detach().numpy())

    # Test 13
    a13 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    b13 = torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.float32)
    results.append((torch.exp(a13) * torch.log(b13) - torch.pow(a13, 0.5)).detach().numpy())

    # Test 14
    import torch.nn.functional as F
    a14 = torch.tensor([[-2.0, -1.0], [1.0, 2.0]], dtype=torch.float32)
    results.append(F.gelu(a14, approximate='tanh').detach().numpy())

    return results

if __name__ == "__main__":
    # If run directly, still print for debugging
    for res in run():
        dump_torch_tensor(torch.from_numpy(res))
