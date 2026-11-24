#!/usr/bin/env python3
"""2-layer ARA intervention sweep: Layer 29 + one other layer L.

Tests Layer 29 combined with each candidate layer L (default: 21..41 excluding 29)
using coder3101 ARA weights, with ARA module I/O disk caching.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from peft import PeftModel
import torch

from heretic.config import QuantizationMethod, Settings
from heretic.evaluator import Evaluator
from heretic.model import Model
from heretic.modifiers.ara import ARA, Parameters as ARAParameters, Settings as ARASettings
from heretic.plugin import Context


CODER3101_WEIGHTS = {
    "preserve_good_behavior_weight": 0.3873,
    "steer_bad_behavior_weight": 0.0003,
    "overcorrect_relative_weight": 0.8555,
    "neighbor_count": 14,
}


def save_ara_cache(modifier: ARA, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "good_module_io": modifier.good_module_io,
            "bad_module_io": modifier.bad_module_io,
        },
        cache_path,
    )
    size_mb = cache_path.stat().st_size / (1024 * 1024)
    print(f"Saved ARA cache to {cache_path} ({size_mb:.1f} MB)", flush=True)


def load_ara_cache(modifier: ARA, cache_path: Path) -> None:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    modifier.good_module_io = payload["good_module_io"]
    modifier.bad_module_io = payload["bad_module_io"]
    print(f"Loaded ARA cache from {cache_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-layer", type=int, default=29)
    parser.add_argument("--start-layer", type=int, default=21)
    parser.add_argument("--end-layer", type=int, default=42, help="Exclusive upper bound")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cache-path", default="/content/cache/gemma4_e4b_bnb4bit_seed42_ara.pt")
    parser.add_argument("--output-json", default="/content/layer29_plus1/sweep_results.json")
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if out_path.exists():
        for rec in json.loads(out_path.read_text(encoding="utf-8")):
            existing[rec["other_layer"]] = rec

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
        print(f"ARA init took {time.perf_counter() - t0:.1f}s", flush=True)
        save_ara_cache(modifier, cache_path)

    if not isinstance(model.model, PeftModel):
        model.apply_lora(modifier.settings.lora_rank)

    results = list(existing.values())
    first_modify = True

    for other_layer in range(args.start_layer, args.end_layer):
        if other_layer == args.base_layer:
            continue
        if other_layer in existing:
            print(f"[Pair ({args.base_layer}, {other_layer})] Already evaluated, skipping.", flush=True)
            continue

        t0 = time.perf_counter()
        if not first_modify:
            modifier.reset_model(ctx)
        first_modify = False

        for layer_idx in (args.base_layer, other_layer):
            params = ARAParameters(
                start_layer_index=layer_idx,
                end_layer_index=layer_idx + 1,
                **CODER3101_WEIGHTS,
            )
            modifier.modify_model(ctx, params)

        score_pairs = evaluator.get_scores()
        scores = {name: float(score.value) for name, score in score_pairs}
        elapsed = round(time.perf_counter() - t0, 2)

        rec = {
            "layers": [args.base_layer, other_layer],
            "other_layer": other_layer,
            "weights": CODER3101_WEIGHTS,
            "scores": scores,
            "elapsed_sec": elapsed,
        }
        results.append(rec)
        results.sort(key=lambda r: r["other_layer"])
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[Pair ({args.base_layer}, {other_layer:2d})] ({elapsed}s) "
            f"Refusals={scores.get('Refusals')} KL={scores.get('KL divergence'):.4f}",
            flush=True,
        )

    print("2-layer sweep completed!", flush=True)


if __name__ == "__main__":
    main()
