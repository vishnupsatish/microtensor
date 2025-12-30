import torch
from torch_utils import dump_torch_tensor

def run():
    results = []

    # Test 1: Simple 2D Matrix Multiplication
    a1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    b1 = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=torch.float32)
    results.append(torch.matmul(a1, b1).numpy())

    # Test 2: Matrix-Vector Multiplication
    a2 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    b2 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    results.append(torch.matmul(a2, b2).numpy())

    # Test 3: Batched Matmul with Broadcasting
    a3 = torch.ones((2, 2, 2), dtype=torch.float32)
    b3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    results.append(torch.matmul(a3, b3).numpy())

    return results

if __name__ == "__main__":
    for res in run():
        dump_torch_tensor(torch.from_numpy(res))

