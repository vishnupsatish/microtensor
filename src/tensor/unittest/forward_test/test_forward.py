import torch

def dump_torch_tensor(t):
    print(t.ndim)
    print(" ".join(map(str, t.shape)))
    for val in t.flatten():
        print(f"{val.item():.6f}", end=" ")
    print()

if __name__ == "__main__":
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    b = torch.tensor([1.0, 2.0], dtype=torch.float32)
    c = a + b
    dump_torch_tensor(c)