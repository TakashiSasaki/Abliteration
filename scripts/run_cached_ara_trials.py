#!/usr/bin/env python3
"""Run multiple Heretic ARA trials with disk caching of module I/O and scorer baselines.

Avoids recomputing the ~12-minute 800-prompt ARA module I/O capture and baseline
probability distributions across repeated experiments or local hyperparameter sweeps.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from heretic.config import QuantizationMethod, Settings
from heretic.evaluator import Evaluator
from heretic.model import Model
from heretic.modifiers.ara import ARA, Parameters as ARAParameters, Settings as ARASettings
from heretic.plugin import Context


DEFAULT_TRIALS = [
    {
        "name": "layer25_coder3101",
        "start_layer_index": 25,
        "end_layer_index": 26,
        "preserve_good_behavior_weight": 0.3873,
        "steer_bad_behavior_weight": 0.0003,
        "overcorrect_relative_weight": 0.8555,
        "neighbor_count": 14,
    },
]


def save_ara_cache(modifier: ARA, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "good_module_io": modifier.good_module_io,
        "bad_module_io": modifier.bad_module_io,
    }
    torch.save(payload, cache_path)
    size_mb = cache_path.stat().st_size / (1024 * 1024)
    print(f"Saved ARA module I/O cache to {cache_path} ({size_mb:.1f} MB)")


def load_ara_cache(modifier: ARA, cache_path: Path) -> None:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    modifier.good_module_io = payload["good_module_io"]
    modifier.bad_module_io = payload["bad_module_io"]
    print(f"Loaded ARA module I/O cache from {cache_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", default="/content/cache/gemma4_e4b_bnb4bit_seed42_ara.pt")
    parser.add_argument("--trials-json", default=None, help="Path to JSON list of trial dicts")
    parser.add_argument("--output-json", default="/content/layer25_sweep_results.json")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    trials = DEFAULT_TRIALS
    if args.trials_json:
        trials = json.loads(Path(args.trials_json).read_text(encoding="utf-8"))

    settings = Settings(
        model="google/gemma-4-E4B-it",
        quantization=QuantizationMethod.BNB_4BIT,
        batch_size=args.batch_size,
        seed=42,
    )
    model = Model(settings)
    ctx = Context(settings=settings, model=model)

    evaluator = Evaluator(settings, model)
    modifier = ARA(heretic_settings=settings, settings=ARASettings())

    cache_path = Path(args.cache_path)
    if cache_path.exists():
        load_ara_cache(modifier, cache_path)
    else:
        t0 = time.perf_counter()
        modifier.init(ctx)
        print(f"ARA init took {time.perf_counter() - t0:.1f}s")
        save_ara_cache(modifier, cache_path)

    from peft import PeftModel
    if not isinstance(model.model, PeftModel):
        model.apply_lora(modifier.settings.lora_rank)

    results = []
    for i, t in enumerate(trials):
        t_start = time.perf_counter()
        params = ARAParameters(
            start_layer_index=t["start_layer_index"],
            end_layer_index=t["end_layer_index"],
            preserve_good_behavior_weight=t["preserve_good_behavior_weight"],
            steer_bad_behavior_weight=t["steer_bad_behavior_weight"],
            overcorrect_relative_weight=t["overcorrect_relative_weight"],
            neighbor_count=t["neighbor_count"],
        )
        if i > 0:
            modifier.reset_model(ctx)
        modifier.modify_model(ctx, params)

        score_pairs = evaluator.get_scores()
        scores = {name: float(score.value) for name, score in score_pairs}

        elapsed = round(time.perf_counter() - t_start, 2)
        rec = {
            "name": t.get("name", f"trial_{i}"),
            "params": params.to_dict(),
            "scores": scores,
            "elapsed_sec": elapsed,
        }
        results.append(rec)
        print(f"[{rec['name']}] ({elapsed}s) -> {scores}")

    out_path = Path(args.output_json)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
