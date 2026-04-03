#!/bin/bash
#SBATCH -A mfk-gsr2
#SBATCH -p ISP
#SBATCH -J st-deconv-cpu
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 08:00:00
#SBATCH -o outputs/logs/%x-%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/net/8k3/e0fs01/users/xj62kv/260402_spatial-cell-deconvolution}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
DEVICE="cpu"

cd "$PROJECT_DIR"

mkdir -p outputs/logs outputs/models outputs/metrics outputs/predictions outputs/figures outputs/data

echo "[$(date)] project dir: $PROJECT_DIR"
echo "[$(date)] python: $PYTHON_BIN"
echo "[$(date)] device: $DEVICE"

echo "[$(date)] inspect dataset"
"$PYTHON_BIN" scripts/inspect_data.py --dataset human_lymph_node

echo "[$(date)] train mlp"
"$PYTHON_BIN" scripts/train_model.py \
  --dataset human_lymph_node \
  --model mlp \
  --epochs 10 \
  --num-spots 1000 \
  --max-sc-cells 6000 \
  --device "$DEVICE"

echo "[$(date)] train spatial gcn"
"$PYTHON_BIN" scripts/train_model.py \
  --dataset human_lymph_node \
  --model spatial_gcn \
  --epochs 10 \
  --num-spots 1000 \
  --max-sc-cells 6000 \
  --device "$DEVICE"

echo "[$(date)] inference mlp"
"$PYTHON_BIN" scripts/run_inference.py \
  --dataset human_lymph_node \
  --checkpoint outputs/models/human_lymph_node_mlp.pt \
  --max-sc-cells 6000 \
  --device "$DEVICE"

echo "[$(date)] inference spatial gcn"
"$PYTHON_BIN" scripts/run_inference.py \
  --dataset human_lymph_node \
  --checkpoint outputs/models/human_lymph_node_spatial_gcn.pt \
  --max-sc-cells 6000 \
  --device "$DEVICE"

echo "[$(date)] done"
