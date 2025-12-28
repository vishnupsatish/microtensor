def dump_torch_tensor(t):
    # Matches the format: Rank, Shape, Data
    print(t.ndim)
    print(" ".join(map(str, t.shape)))
    for val in t.flatten():
        print(f"{val.item():.6f}", end=" ")
    print()

