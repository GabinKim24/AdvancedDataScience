# CheXpert 5-Label Classification: DenseNet121 vs DINOv3

Frozen-feature linear/MLP probes and last-block fine-tuning for the CheXpert
5-label chest X-ray task (Atelectasis, Cardiomegaly, Consolidation, Edema,
Pleural Effusion), comparing an ImageNet-pretrained **DenseNet121** against a
self-supervised **DINOv3 ViT-B/16** backbone.


## Repository structure

```text
chexpert_data.py                  # Dataset, transforms, U-Zeros labels, TARGET_LABELS
make_intra_split.py               # Patient-wise train / intra-val split
apply_uzeros_labels.py            # U-Zeros relabeling of split CSVs
count_label_distribution.py       # 5-label value-distribution stats CSV
extract_densenet121_features.py   # Frozen DenseNet121 (ImageNet) features
extract_dinov3_features.py        # Frozen DINOv3 features (cls / mean_patch / mean_max_patch)
train_linear_head.py              # Linear probe on cached features
train_mlp_head.py                 # MLP probe on cached features
evaluate_heads.py                 # Evaluate frozen-head checkpoints (metrics, ROC, confusion, t-SNE/UMAP)
plot_feature_umap.py              # UMAP figures from feature files
finetune_dinov3_last_block.py     # DINOv3 last-block fine-tune
finetune_densenet121_last_block.py# DenseNet121 denseblock4 fine-tune
evaluate_finetuned_models.py      # Evaluate fine-tuned image checkpoints
```

## Data layout

Place the CheXpert CSVs and images under `data/` (the dataset is **not**
included):

```text
data/train.csv
data/valid.csv
data/valid/ , train/ ...        # CheXpert(-v1.0-small) image folders
```

## Reproducing the main results

### 1. Splits and U-Zeros labels

```bash
python make_intra_split.py \
  --input data/train.csv \
  --train-output data/train_internal.csv \
  --intraval-output data/intraval_internal.csv \
  --train-ratio 0.9 --seed 42

python apply_uzeros_labels.py \
  --splits data/train_internal.csv data/intraval_internal.csv data/valid.csv \
  --output-dir data/processed
```

### 2. Frozen feature extraction

```bash
# DenseNet121 (ImageNet)
python extract_densenet121_features.py --device cuda --batch-size 64 --num-workers 4

# DINOv3 ViT-B/16, mean-patch (main)
python extract_dinov3_features.py \
  --model-name facebook/dinov3-vitb16-pretrain-lvd1689m \
  --feature-type mean_patch --device cuda --batch-size 16 --num-workers 4

# DINOv3 ablation variants
python extract_dinov3_features.py --model-name facebook/dinov3-vitb16-pretrain-lvd1689m \
  --feature-type cls --device cuda --batch-size 16 --num-workers 4
python extract_dinov3_features.py --model-name facebook/dinov3-vitb16-pretrain-lvd1689m \
  --feature-type mean_max_patch --device cuda --batch-size 16 --num-workers 4
```

### 3. Train frozen probes

DenseNet121:

```bash
python train_linear_head.py \
  --train-features features/train_internal_uzeros_densenet121_features.pt \
  --val-features   features/intraval_internal_uzeros_densenet121_features.pt \
  --test-features  features/valid_uzeros_densenet121_features.pt \
  --device cuda --epochs 30 --patience 5 --run-name linear_densenet121_imagenet

python train_mlp_head.py \
  --train-features features/train_internal_uzeros_densenet121_features.pt \
  --val-features   features/intraval_internal_uzeros_densenet121_features.pt \
  --test-features  features/valid_uzeros_densenet121_features.pt \
  --device cuda --epochs 30 --patience 5 --hidden-dim 512 --dropout 0.3 \
  --run-name mlp_densenet121
```

DINOv3 mean-patch (same trainers, DINOv3 features):

```bash
python train_linear_head.py \
  --train-features features/train_internal_uzeros_facebook_dinov3_vitb16_pretrain_lvd1689m_mean_patch_features.pt \
  --val-features   features/intraval_internal_uzeros_facebook_dinov3_vitb16_pretrain_lvd1689m_mean_patch_features.pt \
  --test-features  features/valid_uzeros_facebook_dinov3_vitb16_pretrain_lvd1689m_mean_patch_features.pt \
  --device cuda --epochs 30 --patience 5 --run-name linear_dinov3_vitb_mean_patch

python train_mlp_head.py \
  --train-features features/train_internal_uzeros_facebook_dinov3_vitb16_pretrain_lvd1689m_mean_patch_features.pt \
  --val-features   features/intraval_internal_uzeros_facebook_dinov3_vitb16_pretrain_lvd1689m_mean_patch_features.pt \
  --test-features  features/valid_uzeros_facebook_dinov3_vitb16_pretrain_lvd1689m_mean_patch_features.pt \
  --device cuda --epochs 30 --patience 5 --hidden-dim 512 --dropout 0.3 \
  --run-name mlp_dinov3_vitb_mean_patch
```

### 4. Evaluate frozen probes

```bash
python evaluate_heads.py \
  --checkpoint checkpoints/linear_densenet121_imagenet_best.pt \
  --features features/valid_uzeros_densenet121_features.pt \
  --output-dir evaluation --run-name densenet121_linear_valid --device cuda

python evaluate_heads.py \
  --checkpoint checkpoints/mlp_densenet121_best.pt \
  --features features/valid_uzeros_densenet121_features.pt \
  --output-dir evaluation --run-name densenet121_mlp_valid --device cuda

python evaluate_heads.py \
  --checkpoint checkpoints/linear_dinov3_vitb_mean_patch_best.pt \
  --features features/valid_uzeros_facebook_dinov3_vitb16_pretrain_lvd1689m_mean_patch_features.pt \
  --output-dir evaluation --run-name dinov3_vitb_mean_patch_linear_valid --device cuda

python evaluate_heads.py \
  --checkpoint checkpoints/mlp_dinov3_vitb_mean_patch_best.pt \
  --features features/valid_uzeros_facebook_dinov3_vitb16_pretrain_lvd1689m_mean_patch_features.pt \
  --output-dir evaluation --run-name dinov3_vitb_mean_patch_mlp_valid --device cuda
```

### 5. Last-block fine-tuning

Defaults already match the reported runs.

```bash
# DenseNet121: denseblock4 + norm5 (run name densenet121_denseblock4_mlp_30ep_bs16_w4_h512_drop01)
python finetune_densenet121_last_block.py --device cuda \
  --batch-size 16 --num-workers 4 --epochs 30 --patience 5 \
  --hidden-dim 512 --dropout 0.1 --backbone-lr 1e-5 --head-lr 1e-4

# DINOv3: last transformer block (run name dinov3_last_block_mean_patch_mlp_30ep_bs16_w4_h512_drop01)
python finetune_dinov3_last_block.py --device cuda \
  --feature-type mean_patch --head-type mlp \
  --batch-size 16 --num-workers 4 --epochs 30 --patience 5 \
  --hidden-dim 512 --dropout 0.1 --backbone-lr 1e-5 --head-lr 1e-4
```

### 6. Evaluate fine-tuned checkpoints

```bash
python evaluate_finetuned_models.py \
  --checkpoint checkpoints/densenet121_denseblock4_mlp_30ep_bs16_w4_h512_drop01_best.pt \
  --eval-csv data/processed/valid_uzeros.csv --data-root data --device cuda

python evaluate_finetuned_models.py \
  --checkpoint checkpoints/dinov3_last_block_mean_patch_mlp_30ep_bs16_w4_h512_drop01_best.pt \
  --eval-csv data/processed/valid_uzeros.csv --data-root data --device cuda
```

### 7. UMAP feature visualization

```bash
python plot_feature_umap.py \
  --feature-files \
    features/train_internal_uzeros_densenet121_features.pt \
    features/train_internal_uzeros_facebook_dinov3_vitb16_pretrain_lvd1689m_mean_patch_features.pt \
  --output-dir umap_outputs/train_mean_patch_primary \
  --color-by primary --max-samples 10000
```

## Main results (validation set)

| Model | Head | macro AUROC | macro F1 |
|-------|------|-------------|----------|
| DenseNet121 (frozen) | Linear | 0.8191 | 0.5750 |
| DenseNet121 (frozen) | MLP | 0.8229 | 0.5880 |
| DINOv3 ViT-B mean-patch (frozen) | Linear | 0.8088 | 0.5663 |
| DINOv3 ViT-B mean-patch (frozen) | MLP | 0.8155 | 0.5727 |
| DenseNet121 (fine-tuned, denseblock4) | MLP | 0.8286 | 0.6085 |
| DINOv3 ViT-B mean-patch (fine-tuned, last block) | MLP | 0.8488 | 0.6096 |

## References

- M. Baharoon et al. *Towards General Purpose Vision Foundation Models for
  Medical Image Analysis: An Experimental Study of DINOv2 on Radiology
  Benchmarks.* arXiv:2312.02366. https://github.com/MohammedSB/DINOv2ForRadiology
- Stomper10/CheXpert. https://github.com/Stomper10/CheXpert
- J. Irvin et al. *CheXpert: A Large Chest Radiograph Dataset with Uncertainty
  Labels and Expert Comparison.* AAAI 2019.
- Oquab et al. *DINOv2*; DINOv3 backbones via Hugging Face `transformers`.
