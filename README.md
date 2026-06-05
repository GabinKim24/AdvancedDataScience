# CheXpert DINOv3 vs DenseNet121

This repository contains the source code used for the final project experiments on
5-label CheXpert chest X-ray classification.

The code is organized as a reusable `chexpert/` library package with thin
entrypoint scripts in `scripts/` — the layout used by `mlmed/torchxrayvision`
(package + `scripts/`) and `facebookresearch/dinov2` (`eval/` library modules
driven by per-task entrypoints):

```text
.
├── requirements.txt
├── chexpert/                    # importable library (no CLIs here)
│   ├── data.py                  # Dataset, transforms, TARGET_LABELS (single source of truth)
│   ├── prepare.py               # patient-wise split + U-Zeros relabeling
│   ├── models.py                # heads, DenseNet121/DINOv3 classifiers, backbone loaders, pooling, unfreezing
│   ├── losses.py                # BCE pos_weight + focal loss
│   ├── metrics.py               # AUROC / F1 / confusion / ROC (per-path metric variants)
│   ├── engine.py                # train/eval loops, seeding, standardization, checkpoint & CSV I/O
│   ├── features.py              # frozen feature extraction
│   └── plotting.py              # ROC, confusion matrices, metric bars, t-SNE/UMAP figures
└── scripts/                     # entrypoints (argparse + wiring)
    ├── check_dataset.py         # sanity-check a CSV against its images
    ├── prepare_data.py          # patient-wise splits + U-Zeros labels
    ├── extract_features.py      # frozen feature extraction
    ├── train_head.py            # train frozen linear/MLP heads
    ├── finetune.py              # fine-tune last blocks
    ├── evaluate.py              # evaluate frozen-head or fine-tuned checkpoints
    ├── plot_umap.py             # UMAP visualizations
    └── export_chexlocalize_case_panels.py  # export CheXlocalize case panels
```

## Code References

The implementation was written for this project and follows common PyTorch
training/evaluation patterns for CheXpert-style multi-label chest X-ray
classification. The organization and workflow were informed by:

- `mlmed/torchxrayvision`: chest X-ray dataset/model utilities and the package +
  `scripts/` training-entrypoint layout (`datasets.py` label lists, transforms,
  `train_epoch`/`valid_test_epoch` engine, per-class AUROC).
- `jfhealthcare/Chexpert`: directory-per-concern CheXpert classifier layout with
  a swappable backbone behind a single classifier wrapper.
- `facebookresearch/dinov2` and `facebookresearch/dino`: DINO feature-extraction
  and linear-probe evaluation structure (frozen backbone → head, with the
  fine-tune path reusing the same head and unfreezing the last blocks).
- Hugging Face Transformers documentation for loading DINO-family vision backbones.

No source files were copied verbatim from these repositories.

## Setup

```bash
pip install -r requirements.txt
```

The code expects CheXpert CSV files and images under `data/` by default. The
dataset itself is not included in this repository.

## Reproducing Main Results

Prepare patient-wise splits and U-Zeros labels:

```bash
python scripts/prepare_data.py all
```

Extract frozen features:

```bash
python scripts/extract_features.py --model densenet121
python scripts/extract_features.py --model dinov3 --model-name facebook/dinov3-vitb16-pretrain-lvd1689m --feature-type mean_patch
```

Train frozen linear/MLP heads:

```bash
python scripts/train_head.py --head-type linear --train-features features/train_internal_uzeros_densenet121_features.pt --val-features features/intraval_internal_uzeros_densenet121_features.pt
python scripts/train_head.py --head-type mlp --train-features features/train_internal_uzeros_facebook_dinov3_vitb16_pretrain_lvd1689m_mean_patch_features.pt --val-features features/intraval_internal_uzeros_facebook_dinov3_vitb16_pretrain_lvd1689m_mean_patch_features.pt
```

Fine-tune last blocks:

```bash
python scripts/finetune.py --model densenet121
python scripts/finetune.py --model dinov3 --model-name facebook/dinov3-vitb16-pretrain-lvd1689m --feature-type mean_patch
```

Evaluate checkpoints:

```bash
python scripts/evaluate.py --checkpoint checkpoints/mlp_densenet121_best.pt --features features/valid_uzeros_densenet121_features.pt
python scripts/evaluate.py --checkpoint checkpoints/dinov3_last_block_mean_patch_mlp_30ep_bs16_w4_h512_drop01_best.pt --eval-csv data/processed/valid_uzeros.csv
```
