#!/usr/bin/env python3
"""Quick CUDA sanity check for local WSL / RTX 5080."""
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    x = torch.zeros(1, device="cuda")
    print("cuda tensor ok:", x.device)
