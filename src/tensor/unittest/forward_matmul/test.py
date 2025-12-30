import torch
from torch_utils import dump_torch_tensor

def run():
    results = []

    # Test 1
    a1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    b1 = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=torch.float32)
    results.append(torch.matmul(a1, b1).numpy())

    # Test 2
    a2 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    b2 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    results.append(torch.matmul(a2, b2).numpy())

    # Test 3
    a3 = torch.ones((2, 2, 2), dtype=torch.float32)
    b3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    results.append(torch.matmul(a3, b3).numpy())

    # Test 4
    a4 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    b4 = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32)
    results.append(torch.matmul(a4, b4).numpy())

    # Test 5
    a5 = torch.tensor([1.0, 2.0], dtype=torch.float32)
    b5 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    results.append(torch.matmul(a5, b5).numpy())

    # Test 6
    a6 = torch.ones((2, 1, 4, 3), dtype=torch.float32)
    b6 = torch.ones((1, 3, 3, 2), dtype=torch.float32) * 2.0
    results.append(torch.matmul(a6, b6).numpy())

    return results

if __name__ == "__main__":
    for res in run():
        dump_torch_tensor(torch.from_numpy(res))

