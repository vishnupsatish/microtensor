import torch
import torch.nn.functional as F
from torch_utils import dump_torch_tensor

def run():
    results = []

    # Test 1: MSE Forward
    a1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    b1 = torch.tensor([[1.5, 1.5], [3.5, 3.5]], dtype=torch.float32)
    results.append(F.mse_loss(a1, b1).numpy())

    # Test 2: MSE Backward
    a2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True)
    b2 = torch.tensor([[1.5, 1.5], [3.5, 3.5]], dtype=torch.float32)
    loss2 = F.mse_loss(a2, b2)
    loss2.backward()
    results.append(a2.grad.numpy())

    # Test 3: Cross Entropy Forward (Basic 2D)
    logits3 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    targets3 = torch.tensor([2, 1], dtype=torch.long)
    results.append(F.cross_entropy(logits3, targets3).numpy())

    # Test 4: Cross Entropy Backward (Basic 2D)
    logits4 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32, requires_grad=True)
    targets4 = torch.tensor([2, 1], dtype=torch.long)
    loss4 = F.cross_entropy(logits4, targets4)
    loss4.backward()
    results.append(logits4.grad.numpy())

    # Test 5: Cross Entropy Forward (Transformer Style 3D)
    # PyTorch CrossEntropy expects (N, C, ...) or (N, C)
    # Our implementation expects (N, C) after reshaping.
    # In 3D (B, T, C), PyTorch expects (B, C, T).
    logits5 = torch.tensor([[[0.1, 0.2, 0.7], [0.9, 0.05, 0.05]]], dtype=torch.float32)
    targets5 = torch.tensor([[2, 0]], dtype=torch.long)
    # We reshape to (B*T, C) to match our C++ implementation's behavior
    res5 = F.cross_entropy(logits5.view(-1, 3), targets5.view(-1))
    results.append(res5.numpy())

    # Test 6: Cross Entropy Backward (Transformer Style 3D)
    logits6 = torch.tensor([[[0.1, 0.2, 0.7], [0.9, 0.05, 0.05]]], dtype=torch.float32, requires_grad=True)
    targets6 = torch.tensor([[2, 0]], dtype=torch.long)
    loss6 = F.cross_entropy(logits6.view(-1, 3), targets6.view(-1))
    loss6.backward()
    results.append(logits6.grad.numpy())

    return results

if __name__ == "__main__":
    for res in run():
        dump_torch_tensor(torch.from_numpy(res))
