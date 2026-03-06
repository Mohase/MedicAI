"""
Quick sanity check: one forward pass through FDR-TransUNet.
"""
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

print("Loading PyTorch...")
import torch

print("Loading model...")
from model import FDRTransUNet

def main():

    print("Building FDR-TransUNet...")
    model = FDRTransUNet(in_channels=1, input_h=256, input_w=256)
    model.eval()
    x = torch.randn(2, 1, 256 ,256)

    print("Running forward pass...")
    with torch.no_grad():
        combined, logits_a, logits_b = model(x)
    print("Shapes: ", combined.shape, logits_a.shape, logits_b.shape)
    assert combined.shape == (2, 1, 256, 256), f"Expected (2,1,256,256), got {combined.shape}"
    print("OK: forward pass succeeded!")

if __name__ == "__main__":
    main()