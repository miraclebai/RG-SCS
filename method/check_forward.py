"""Run a CPU forward pass with random weights and normalized-shape IQ tensors."""
import json
from pathlib import Path

import torch

from model import RGSCS


def main():
    config = json.loads(Path(__file__).with_name("config.json").read_text())
    torch.manual_seed(0)
    torch.set_num_threads(1)
    model = RGSCS(config).eval()
    x = torch.randn(1, 256, 16, dtype=torch.complex64)
    left, right = torch.randn(2, 1, 256, 8, dtype=torch.complex64)
    with torch.no_grad():
        prediction = model(x, left, right)
    output = prediction["s_hat"]
    residual = prediction["solve_residual"].max().item()
    if output.shape != x.shape or not torch.isfinite(output).all() or not residual < 5e-5:
        raise RuntimeError(f"Forward check failed (residual {residual:g})")
    print(f"Output shape {tuple(output.shape)}, relative residual {residual:.2e}")


if __name__ == "__main__":
    main()
