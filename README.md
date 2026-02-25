# PyTorch Classification Experiment Suite

Baseline experiment framework for multiclass image classification with:
- Model zoo wrappers (`resnet18`, `resnet50`, `vgg16`, `mobilenetv2`, `shufflenetv2`, `efficientnet_b0`, `efficientnet_b1`)
- Dataset loaders (`cifar10`, `cifar100`, `food101`, `stanford_cars`, `oxford_pets`)
- Unified trainer with checkpointing and reproducible train/val split
- Primary + efficiency metrics
- Grid runner over model x dataset x seed

## Install

```bash
pip install -r requirements.txt
```

## Run a single experiment

```bash
python -m experiments.run_one \
  --model resnet18 \
  --dataset cifar10 \
  --seed 0 \
  --pretrained true \
  --img-size 224 \
  --epochs 30 \
  --save-weight-models kannet*
```

## Run the suite

```bash
python -m experiments.suite \
  --pretrained true \
  --img-size 224 \
  --epochs 30 \
  --seeds 0,1,2 \
  --save-weight-models kannet*
```

Disable tqdm/IPS in training loops:

```bash
python -m experiments.suite --no-tqdm --no-ips
```

Override model/dataset lists:

```bash
python -m experiments.suite \
  --models resnet18,mobilenetv2,shufflenetv2 \
  --datasets cifar10,cifar100 \
  --seeds 0,1,2
```

## Recorded metrics

- Primary: `test_top1`, `test_top5`, `test_roc_auc_macro`
- Efficiency: `param_count`, `flops_gmacs`, `throughput_images_per_sec`
- Runtime: `total_train_time_sec`, per-epoch `epoch_time_sec`

## Outputs and reproducibility

Per run artifacts are written to:

`outputs/runs/{dataset}/{model}/seed{seed}/`

Files:
- `config.json`
- `history.json`
- `best.pt`
- `result.json`

Optional exported model weights (separate from run results):
- `outputs/weights/{dataset}/{model}/seed{seed}/model_state_dict.pt`
- Controlled with `--save-weight-models` (default: `kannet*`) and `--weights-root`.

Suite outputs:
- `outputs/runs/suite_results.json`
- `outputs/runs/summary.csv` (mean/std across seeds for each dataset+model pair)

Reproducibility:
- Uses `training.utils.set_seed(seed, deterministic=True)` by default.
- Validation split is deterministic from the provided seed.

## Validate model wiring

Run full local smoke validation of all models in `MODEL_REGISTRY`:

```bash
python scripts/validate_models.py
```

With backward smoke check:

```bash
python scripts/validate_models.py --backward
```

With throughput (GPU) and AMP:

```bash
python scripts/validate_models.py --throughput --amp
```

With KAN parameter budget checks:

```bash
python scripts/validate_models.py --check-kan-budgets
```

## Dataset Validation

Validate all datasets in `DATASET_REGISTRY`:

```bash
python scripts/validate_datasets.py
```

Include optional forward checks:

```bash
python scripts/validate_datasets.py --forward
```

Specify model for forward compatibility test:

```bash
python scripts/validate_datasets.py --forward --model resnet18
```

Checks include dataloader construction, sample/batch shape and dtype, label range, finite values, loader iteration timing, and optional model forward compatibility.

Download/cache all datasets before training:

```bash
python scripts/download_datasets.py --data-root data
```

Download only selected datasets:

```bash
python scripts/download_datasets.py --datasets cifar10,cifar100,food101
```

## ImageNet-1K pretraining for KAN models

Expected ImageNet folder layout:

```text
/path/to/imagenet/
  train/<class_name>/*.jpeg
  val/<class_name>/*.jpeg
```

Pretrain one KAN model:

```bash
python -m experiments.pretrain_imagenet \
  --model kan_resnet18_tiny \
  --data-root /path/to/imagenet
```

Disable progress bars or images/sec in pretraining:

```bash
python -m experiments.pretrain_imagenet \
  --model kan_resnet18_tiny \
  --data-root /path/to/imagenet \
  --no-tqdm --no-ips
```

Run pretraining suite:

```bash
python -m experiments.pretrain_suite_imagenet \
  --data-root /path/to/imagenet \
  --seeds 0,1
```

Train all KAN ImageNet variants and export pretrained weights:

```bash
python -m experiments.pretrain_suite_imagenet \
  --data-root /path/to/imagenet \
  --seeds 0,1
```

Outputs:
- Run checkpoints/history: `outputs/imagenet_pretrain/<model>/seed<seed>/`
- Exported pretrained weights: `weights/imagenet/<model>_imagenet1k_seed<seed>.pt`
- Suite summary: `outputs/imagenet_pretrain/suite_summary.csv`

Optional ImageNet smoke test:

```bash
python scripts/smoke_imagenet1k.py --data-root /path/to/imagenet
python scripts/smoke_imagenet1k.py --data-root /path/to/imagenet --run-debug-pretrain
```
