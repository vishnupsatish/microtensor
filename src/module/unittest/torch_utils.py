def dump_torch_tensor(t):
    # Used for differential testing.
    # Format:
    # rank
    # shape[0] shape[1] ... shape[rank-1]
    # data[0] data[1] ... data[total_elements-1]
    print(t.ndim)
    print(" ".join(map(str, t.shape)))
    for val in t.flatten():
        print(f"{val.item():.6f}", end=" ")
    print()
