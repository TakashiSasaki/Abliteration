#!/usr/bin/env python3
"""Single-layer (layer 25) ARA abliteration experiment for google/gemma-4-E4B-it."""
import os
import subprocess
from pathlib import Path

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from heretic.config import QuantizationMethod, Settings

LAYER25_PARAMS = {
    "start_layer_index": 25,
    "end_layer_index": 26,
    "preserve_good_behavior_weight": 0.3873,
    "steer_bad_behavior_weight": 0.0003,
    "overcorrect_relative_weight": 0.8555,
    "neighbor_count": 14,
}


def main():
    ckpt_dir = Path("/content/layer25_exp/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = ckpt_dir / "google--gemma-4-E4B-it.jsonl"
    if ckpt_file.exists():
        ckpt_file.unlink()

    # heretic computes n_trials = settings.n_trials - len(study.trials).
    # Since enqueue_trial() adds 1 WAITING trial (len=1), n_trials=2 runs exactly Trial 0.
    settings = Settings(
        model="google/gemma-4-E4B-it",
        quantization=QuantizationMethod.BNB_4BIT,
        batch_size=128,
        n_trials=2,
        n_startup_trials=0,
        seed=42,
    )

    storage = JournalStorage(JournalFileBackend(str(ckpt_file)))
    study = optuna.create_study(
        study_name="heretic",
        storage=storage,
        directions=["minimize", "minimize"],
    )
    study.set_user_attr("settings", settings.model_dump_json())
    study.set_user_attr("finished", False)
    study.enqueue_trial(LAYER25_PARAMS)

    save_dir = Path("/content/layer25_exp/saved_adapter")
    save_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "heretic",
        "--model", "google/gemma-4-E4B-it",
        "--quantization", "BNB_4BIT",
        "--batch-size", "128",
        "--n-trials", "2",
        "--n-startup-trials", "0",
        "--seed", "42",
        "--study-checkpoint-dir", str(ckpt_dir),
        "--checkpoint-action", "continue",
        "--trial-index", "0",
        "--model-action", "save",
        "--save-directory", str(save_dir),
        "--export-strategy", "ADAPTER",
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
