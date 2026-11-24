#!/usr/bin/env bash
# Reproducible Heretic (ARA) abliteration run for google/gemma-4-E4B-it on Colab T4 (15GB VRAM)
# Based on p-e-w/heretic v2.1.0 and coder3101/gemma-4-E4B-it-heretic reference.
set -euo pipefail

OUT_DIR="${1:-/content/heretic_out}"
N_TRIALS="${N_TRIALS:-20}"
N_STARTUP="${N_STARTUP:-10}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-128}"

mkdir -p "$OUT_DIR"

exec heretic \
  --model google/gemma-4-E4B-it \
  --quantization BNB_4BIT \
  --batch-size "$BATCH_SIZE" \
  --n-trials "$N_TRIALS" \
  --n-startup-trials "$N_STARTUP" \
  --seed "$SEED" \
  --checkpoint-action restart \
  --trial-index 0 \
  --model-action save \
  --save-directory "$OUT_DIR" \
  --export-strategy ADAPTER
