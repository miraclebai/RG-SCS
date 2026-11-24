"""Write a deterministic toy dataset for the training and inference example."""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from radar_common import CORE, PRF, phase_labels


def main(root):
    root.mkdir(parents=True, exist_ok=False)
    (root / "reports").mkdir()
    rng = np.random.default_rng(17)
    time = np.arange(256)[:, None] / PRF
    envelope = np.exp(-0.5 * ((np.arange(16) - 8) / 1.5) ** 2)
    rows = []
    for i, split in enumerate(("train", "train", "test")):
        name = f"sample_{i}"
        frequency = (-125, 150, 90)[i]
        background = (rng.normal(size=(256, 64)) + 1j * rng.normal(size=(256, 64))).astype("c8")
        target = (np.exp(2j * np.pi * frequency * time) * envelope).astype("c8")
        left, right = background[:, :8], background[:, -8:]
        input_path, label_path = f"{name}.npz", f"{name}_label.npz"
        np.savez(root / input_path, x=background[:, CORE] + target, refs_left=left,
                 refs_right=right, refs=np.concatenate((left, right), axis=1),
                 background_raw=background, scale=1.0, metadata_json="{}")
        np.savez(root / label_path, s=target, **phase_labels(target))
        support = np.abs(target) >= np.abs(target).max() / np.sqrt(10)
        scr = 10 * np.log10(np.sum(abs(target[support])**2) / np.sum(abs(background[:, CORE][support])**2))
        rows.append(dict(id=name, split=split, kind="synthetic", input_path=input_path,
                         label_path=label_path, source_id="toy", core_start_one_based=25,
                         scr_support_db=float(scr)))
    with (root / "reports/manifest.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path, help="New dataset directory")
    main(parser.parse_args().output)
