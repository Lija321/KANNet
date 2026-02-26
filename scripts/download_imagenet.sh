#!/usr/bin/env bash
# scripts/download_imagenet.sh
#
# Paste your ImageNet ILSVRC2012 direct download links below, then run:
#   chmod +x scripts/download_imagenet.sh
#   ./scripts/download_imagenet.sh
#
# Output folder (repo-relative):
#   data/imagenet/{train,val}

set -euo pipefail

###############################################################
# PASTE YOUR IMAGENET DOWNLOAD LINKS HERE
###############################################################
TRAIN_URL="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"
VAL_URL="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"

###############################################################
# CONFIG
###############################################################
ROOT="data/imagenet"
TRAIN_TAR="ILSVRC2012_img_train.tar"
VAL_TAR="ILSVRC2012_img_val.tar"

# Use pigz if available (faster extraction). Falls back to plain tar otherwise.
if command -v pigz >/dev/null 2>&1; then
  TAR_GZ_FLAG=(-I "pigz -p $(nproc)")
  echo "[INFO] pigz found -> using parallel extraction with $(nproc) threads"
else
  TAR_GZ_FLAG=()
  echo "[INFO] pigz not found -> using standard tar extraction"
fi

mkdir -p "$ROOT"
cd "$ROOT"

echo "================================================"
echo "ImageNet download + preparation script"
echo "Target folder: $ROOT"
echo "================================================"

###############################################################
# DOWNLOAD (resume supported)
###############################################################
if [[ "$TRAIN_URL" == PASTE_* || "$VAL_URL" == PASTE_* ]]; then
  echo "[ERROR] You must paste TRAIN_URL and VAL_URL at the top of this script."
  exit 1
fi

if [ ! -f "$TRAIN_TAR" ]; then
  echo "==> Downloading TRAIN set..."
  wget -c "$TRAIN_URL" -O "$TRAIN_TAR"
else
  echo "==> TRAIN tar already exists, skipping download: $TRAIN_TAR"
fi

if [ ! -f "$VAL_TAR" ]; then
  echo "==> Downloading VAL set..."
  wget -c "$VAL_URL" -O "$VAL_TAR"
else
  echo "==> VAL tar already exists, skipping download: $VAL_TAR"
fi

###############################################################
# EXTRACT TRAIN
###############################################################
if [ ! -d "train" ]; then
  echo "==> Extracting TRAIN archive into train/ ..."
  mkdir -p train
  tar "${TAR_GZ_FLAG[@]}" -xf "$TRAIN_TAR" -C train
else
  echo "==> train/ already exists, skipping TRAIN archive extraction"
fi

echo "==> Extracting class archives inside train/ (this can take a while)..."
cd train

# Extract each nXXXXXXXX.tar into a folder nXXXXXXXX/
# This is idempotent-ish: if a .tar is gone, we assume it's already extracted.
shopt -s nullglob
for f in *.tar; do
  d="${f%.tar}"
  mkdir -p "$d"
  tar "${TAR_GZ_FLAG[@]}" -xf "$f" -C "$d"
  rm -f "$f"
done
shopt -u nullglob

cd ..

###############################################################
# EXTRACT VAL
###############################################################
if [ ! -d "val" ]; then
  echo "==> Extracting VAL archive into val/ ..."
  mkdir -p val
  tar "${TAR_GZ_FLAG[@]}" -xf "$VAL_TAR" -C val
else
  echo "==> val/ already exists, skipping VAL archive extraction"
fi

###############################################################
# PREPARE VAL STRUCTURE
###############################################################
echo "==> Preparing validation folder structure (val/<class>/ILSVRC...JPEG)..."

VAL_DIR="val"

# If already prepared (class folders exist), skip
if ls "$VAL_DIR"/n*/ >/dev/null 2>&1; then
  echo "==> val/ already appears to be organized into class folders, skipping val prep"
else
  # Download synset labels list used by standard valprep
  LABELS_FILE="imagenet_2012_validation_synset_labels.txt"
  if [ ! -f "$LABELS_FILE" ]; then
    echo "==> Downloading validation synset label list..."
    wget -q https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/imagenet_2012_validation_synset_labels.txt -O "$LABELS_FILE"
  fi

  # Move val images into class folders according to label list order.
  # Expects files named ILSVRC2012_val_00000001.JPEG ... ILSVRC2012_val_00050000.JPEG
  i=1
  while read -r label; do
    mkdir -p "${VAL_DIR}/${label}"
    printf -v img "%08d" "$i"
    src="${VAL_DIR}/ILSVRC2012_val_${img}.JPEG"
    if [ -f "$src" ]; then
      mv "$src" "${VAL_DIR}/${label}/"
    fi
    i=$((i+1))
  done < "$LABELS_FILE"

  # Quick sanity check
  if ! ls "$VAL_DIR"/n*/ >/dev/null 2>&1; then
    echo "[ERROR] Validation set was not organized correctly."
    echo "Check that val/ contains files named ILSVRC2012_val_00000001.JPEG etc."
    exit 2
  fi
fi

###############################################################
# FINAL CHECKS
###############################################################
echo "==> Final sanity checks..."
if [ ! -d "train" ] || [ ! -d "val" ]; then
  echo "[ERROR] Missing train/ or val/ directory."
  exit 3
fi

train_classes=$(find train -maxdepth 1 -type d -name 'n*' | wc -l | tr -d ' ')
val_classes=$(find val -maxdepth 1 -type d -name 'n*' | wc -l | tr -d ' ')

echo "Train class dirs: $train_classes (expected 1000)"
echo "Val class dirs:   $val_classes (expected 1000)"

echo ""
echo "=============================================="
echo "ImageNet is READY."
echo "data/imagenet/train/<class>/..."
echo "data/imagenet/val/<class>/..."
echo "=============================================="