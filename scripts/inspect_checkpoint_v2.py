import torch

try:
    checkpoint = torch.load("model_checkpoint.pt")
    if isinstance(checkpoint, dict):
        print("Keys:", checkpoint.keys())
        if 'encoder' in checkpoint:
            print("--- Encoder ---")
            for k, v in checkpoint['encoder'].items():
                print(f"{k}: {v.shape}")
        if 'head' in checkpoint:
            print("--- Head ---")
            for k, v in checkpoint['head'].items():
                print(f"{k}: {v.shape}")
except Exception as e:
    print(e)
