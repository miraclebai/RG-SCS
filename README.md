# RG-SCS

**Sea Clutter Suppression via Relation-Guided Complementary Constraints**

Sea clutter superimposed on target echoes degrades maritime target detection, making effective suppression essential. However, conventional model-based methods can suffer from mismatch under nonstationarity, while deep learning methods relying solely on learned mappings leave target and clutter relations implicit, which can lead to target attenuation or residual clutter when their spectra overlap. We propose Relation-Guided Sea Clutter Suppression (RG-SCS). It learns target phase evolution from complex echoes and calibrates clutter prediction relations using neighboring references. Complementary constraints couple target consistency of retained components with clutter consistency of removed components through quadratic optimization, balancing target preservation and clutter suppression. Quadratic optimality yields a complex linear system solved by preconditioned conjugate gradient (PCG). Extensive experiments on measured radar data demonstrate effective clutter attenuation and improved target visibility. Source code is available at https://github.com/miraclebai/RG-SCS.

![RG-SCS framework showing target phase estimation, reference-based clutter calibration and joint recovery](assets/framework.png)

## Results on measured echoes

| Input | RG-SCS output |
| :---: | :---: |
| ![Input range–Doppler map](assets/input.png) | ![Filtered range–Doppler map](assets/output.png) |

The figure below compares five baselines with our proposed method on the same measured echo.

![Comparison with Adaptive MTI, k-means SVD, UNet, CV-UNet and CVNN](assets/comparison.png)

## Quick start

From the repository root, install the dependencies with Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place the [weights and measured sample](https://drive.google.com/drive/folders/1kEigFTATnGg4rN3PgM6bga0SSPn2WjKx) in the repository root.

```bash
python method/infer.py --checkpoint best.pt --input real_20221113070041_stare_HH_p000512_r0492.npz --first-range-bin 492 --output-dir inference/measured
```

## Train on the full dataset

Set `DATA_ROOT` to the directory containing [reports/manifest.csv](reports/manifest.csv) and the NPZ files at the paths listed in it. Each sample stores normalized complex IQ arrays in **(pulses, range cells)** order. The input `x` has shape `(256, 16)`; `refs_left` and `refs_right` each have shape `(256, 8)`. The supervised target `s` and background `b` both match `x`. Synthetic examples provide target IQ and phase labels. Background-only examples use a zero target. The loader extracts `b` from the center 16 cells of `background_raw` and divides by the sample scale.

Supply the original NPZ data for this run. The commands below train from scratch and evaluate the resulting checkpoint.

```bash
DATA_ROOT=/path/to/data-root
python method/train.py --data-root "$DATA_ROOT" --epochs 250 --batch-size 768 --device cuda --run-dir runs/full --skip-inference
python method/infer.py --data-root "$DATA_ROOT" --run-dir runs/full --tag best --device cuda --output-dir inference/full
```
