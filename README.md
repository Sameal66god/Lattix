# Localization as Lattice Decision

**Overcoming the Half-Pixel Cliff in Infrared Small Target Detection**

Pre-release draft. Code and pretrained weights will be released with the final version.

## What this is about

A detector's box is usually predicted as four continuous coordinates, but annotations and evaluation live on an integer pixel lattice. For objects spanning only a few pixels, one lattice step is a large fraction of the object itself, and this regression-lattice mismatch shows up as a hard accuracy floor. A 5×5 target shifted by half a pixel already drops to IoU 0.681 — below the 0.75 threshold — and two detectors with very different backbones (FocalNet-B, 0.401 mAP, and YOLO11n, 0.402 mAP) saturate at the same accuracy. The bottleneck is not the features; it is the output paradigm.

Instead of regressing toward the lattice, we decide on it. Each box edge is classified over 33 candidate integer lattice lines inside a CROP=32 window centered on a coarse proposal, and the box is chosen by maximizing its expected COCO credit. The resulting detector, **Lattix**, produces integer boxes by construction, removing the source of the half-pixel cliff rather than shrinking its effect.

Lattix combines three components:

- **EPB** — an evidence purification branch that pulls low-level cues from the backbone and injects them through a zero-initialized residual connection;
- **LatticeHead** — per-edge classification over 33 lattice lines (4 edges × 33), also zero-initialized;
- **BLD** — Bayes lattice decoding: top-3 candidates per edge, expected-credit box selection (T=1.4, β=0.5).

On IRSTD-1K test-201, Lattix reaches 0.4486 mAP (±0.0093, three seeds) and 0.4060 AP75 — gains of +6.37 mAP and +11.24 AP75 over the same-codebase baseline — with about 0.12M extra parameters (3.87M in total) at 49.3 FPS. The gains transfer to SIRSTv2 (+4.87 mAP) and NUDT-SIRST (+5.06 mAP).

## News

- **2026-10** — Pre-release draft. Manuscript under review at *Pattern Recognition*.

## Main results

### IRSTD-1K (test-201, mean over three seeds)

| Method | mAP | AP50 | AP75 |
|---|---|---|---|
| YOLO11n (same protocol) | 0.402 | 0.810 | – |
| Same-codebase baseline | 0.3849 | 0.8360 | 0.2936 |
| NS-FPN (same protocol) | 0.4210 | 0.8630 | 0.3690 |
| **Lattix (ours)** | **0.4486 ± 0.0093** | **0.8593** | **0.4060** |

NS-FPN is re-evaluated under our protocol; details are given in the Table 1 footnotes of the paper.

### Cross-dataset generalization (crop=48, single test evaluation)

| Dataset | Lattix | Gain vs host |
|---|---|---|
| SIRSTv2 | 0.4776 | +4.87 |
| NUDT-SIRST | 0.7184 | +5.06 |

More outputs become directly usable as measurements: near-exact boxes (IoU ≥ 0.9) rise from 17.96% to 27.27%, and boxes whose four edges match the ground truth pixel-for-pixel rise from 10.58% to 13.88%.

## Installation

```bash
conda create -n lattix python=3.10 -y
conda activate lattix
pip install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Developed and tested with PyTorch 2.9.1 (cu126). The evaluation uses pycocotools and faster-coco-eval; training logging uses tensorboard. One caveat: `albumentations==2.0.8` silently drops the `GaussNoise` strength argument, so the repository ships a small patch that restores the intended noise level.

## Dataset preparation

Annotations use the COCO detection format. Boxes are regenerated from masks by 8-connected bounding boxes, and all images are resized to a uniform resolution (bilinear for images, nearest for masks).

```
datasets/
├── irstd1k_det/    # IRSTD-1K: 720 train / 80 val / 201 test (official split)
├── sirstv2_det/    # SIRSTv2: official 512/256/256 split, 512×512
└── nudt_det/       # NUDT-SIRST: 512×512 (×2 upsampling)
```

- IRSTD-1K reuses the official split; test-201 is the only community-shared test set in the field.
- SIRSTv2 is prepared with `make_sirstv2_det.py` (multi-size images unified to 512×512).
- NUDT-SIRST is prepared with `make_nudt_det.py` (×2 upsampling, nearest interpolation). The official split is not public, so we use a fixed local split (663/132/532) and disclose this in the paper.

##培训

```bash
python train.py -c vendor/D-FINE/configs/dfine/irstd1k/dfine_lh_irstd720.yml \
    --use-amp --seed=0 --device cuda:0
```

Main configuration: 720/80, CROP=32, 160 epochs. Reported numbers are the mean over seeds 0, 1 and 3, all trained on a single RTX 4090. Ablation configs (`dfine_epbonly_irstd720.yml`, `dfine_lhonly_irstd720.yml`) are generated from the main one; cross-dataset runs use the crop=48 configs. Training resumes with `-r <checkpoint>` (model, optimizer, scheduler, EMA, scaler and epoch are all restored).

## Evaluation

```bash
# IRSTD-1K: val80 model selection, then a single test-201 run per seed
python diag/t110_lh_eval.py --dataset irstd1k --ckpt output/dfine_lh_irstd1k_n_720/best_stg1.pth
python diag/t110_lh_eval.py --dataset sirstv2 --ckpt <ckpt> --crop 48
python diag/t110_lh_eval.py --dataset nudt   --ckpt <ckpt> --crop 48
```

The val80 split is used only for checkpoint and calibration selection; test-201 is evaluated exactly once per seed. Metrics follow `test_coco_eval_bbox` (mAP / AP50 / AP75 / AP@[.85:.95]).


## License and acknowledgements

License: to be finalized with the public release.

This implementation builds on the D-FINE detection framework and the PaddleDetection HGNetv2 backbone. Datasets: IRSTD-1K (ISNet), SIRSTv2 (OSCAR), and NUDT-SIRST. Training resources were provided by the National University of Defense Technology; this work was funded by the National Natural Science Foundation of China (Grant No. 12505272).
