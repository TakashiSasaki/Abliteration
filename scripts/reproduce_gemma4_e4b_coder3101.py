#!/usr/bin/env python3
"""Reproduce coder3101/gemma-4-E4B-it-heretic parameters using Heretic v2.1.0 (ARA + BNB_4BIT).

Enqueues the published ARA parameters into Optuna JournalStorage and runs Heretic
with --batch-size 128 to skip batch-size probing and save GPU compute credits.
"""
import argparse
import os
import subprocess
from pathlib import Path

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend


CODER3101_PARAMS = {
    "start_layer_index": 21,
    "end_layer_index": 42,
    "preserve_good_behavior_weight": 0.3873,
    "steer_bad_behavior_weight": 0.0003,
    "overcorrect_relative_weight": 0.8555,
    "neighbor_count": 14,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="/content/checkpoints")
    parser.add_argument("--save-dir", default="/content/heretic_reproduced")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--quantization", default="BNB_4BIT")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = ckpt_dir / "google--gemma-4-E4B-it.jsonl"

    if ckpt_file.exists():
        storage = JournalStorage(JournalFileBackend(str(ckpt_file)))
        study = optuna.load_study(study_name="heretic", storage=storage)
        study.enqueue_trial(CODER3101_PARAMS)
        n_trials = len(study.trials)
        ckpt_action = "continue"
    else:
        n_trials = 1
        ckpt_action = "restart"

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        "heretic",
        "--model", "google/gemma-4-E4B-it",
        "--quantization", args.quantization,
        "--batch-size", str(args.batch_size),
        "--n-trials", str(n_trials),
        "--n-startup-trials", "0",
        "--seed", "42",
        "--study-checkpoint-dir", str(ckpt_dir),
        "--checkpoint-action", ckpt_action,
        "--trial-index", "0",
        "--model-action", "save",
        "--save-directory", args.save_dir,
        "--export-strategy", "ADAPTER",
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
