#!/bin/bash
#SBATCH -A mfk-gsr2
#SBATCH -p plcyf-gpu
#SBATCH -J st-deconv-gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o outputs/logs/%x-%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/net/8k3/e0fs01/users/xj62kv/260402_spatial-cell-deconvolution}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
DEVICE="cuda"

cd "$PROJECT_DIR"

mkdir -p outputs/logs outputs/models outputs/metrics outputs/predictions outputs/figures outputs/data

echo "[$(date)] project dir: $PROJECT_DIR"
echo "[$(date)] python: $PYTHON_BIN"
echo "[$(date)] device: $DEVICE"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[$(date)] gpu info"
  nvidia-smi || true
fi

echo "[$(date)] inspect dataset"
"$PYTHON_BIN" scripts/inspect_data.py --dataset human_lymph_node

echo "[$(date)] train mlp"
"$PYTHON_BIN" scripts/train_model.py \
  --dataset human_lymph_node \
  --model mlp \
  --epochs 20 \
  --num-spots 2000 \
  --max-sc-cells 8000 \
  --device "$DEVICE"

echo "[$(date)] train spatial gcn"
"$PYTHON_BIN" scripts/train_model.py \
  --dataset human_lymph_node \
  --model spatial_gcn \
  --epochs 20 \
  --num-spots 2000 \
  --max-sc-cells 8000 \
  --device "$DEVICE"

echo "[$(date)] inference mlp"
"$PYTHON_BIN" scripts/run_inference.py \
  --dataset human_lymph_node \
  --checkpoint outputs/models/human_lymph_node_mlp.pt \
  --max-sc-cells 8000 \
  --device "$DEVICE"

echo "[$(date)] inference spatial gcn"
"$PYTHON_BIN" scripts/run_inference.py \
  --dataset human_lymph_node \
  --checkpoint outputs/models/human_lymph_node_spatial_gcn.pt \
  --max-sc-cells 8000 \
  --device "$DEVICE"

echo "[$(date)] done"
