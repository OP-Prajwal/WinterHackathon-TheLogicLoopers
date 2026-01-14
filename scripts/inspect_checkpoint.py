import torch

try:
    checkpoint = torch.load("model_checkpoint.pt")
    print("Checkpoint type:", type(checkpoint))
    if isinstance(checkpoint, dict):
        print("Keys:", checkpoint.keys())
        for k, v in checkpoint.items():
            if isinstance(v, dict): # State dict likely
                 print(f"--- {k} keys ---")
                 for sk, sv in list(v.items())[:5]:
                     print(f"{sk}: {sv.shape}")
            else:
                print(f"{k}: {v}")
    else:
        print("Checkpoint is not a dict.")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
