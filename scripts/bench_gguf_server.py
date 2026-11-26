#!/usr/bin/env python3
"""Query a running llama-server /v1/chat/completions endpoint for speed, MTP acceptance, and sanity checks."""
import argparse
import json
from pathlib import Path
import urllib.request

DEFAULT_PROMPTS = [
    (
        "en_tech",
        "Explain how HTTPS TLS 1.3 handshakes achieve 1-RTT latency in numbered steps.",
    ),
    (
        "ja_code",
        "Pythonで二分探索を実装し、計算量とエッジケースを日本語で簡潔に解説してください。",
    ),
    (
        "en_reason",
        "A bat and a ball cost 1.10 dollars in total. The bat costs 1.00 dollar more than the ball. "
        "How much does the ball cost? Explain briefly.",
    ),
    (
        "ja_benign_border",
        "小説の執筆用に、中世ヨーロッパの城壁の構造と攻城戦の一般的な戦術を歴史資料として要約してください。",
    ),
]


def query_prompt(base_url: str, prompt_id: str, prompt: str, max_tokens: int, disable_thinking: bool) -> dict:
    """Send a single deterministic completion request and extract timings + response head."""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": "local-gguf",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 42,
    }
    if disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    msg = data["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    return {
        "id": prompt_id,
        "timings": data.get("timings") or {},
        "response_head": content[:200],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark llama-server throughput and MTP draft acceptance.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--disable-thinking", action="store_true", default=True)
    parser.add_argument("--enable-thinking", dest="disable_thinking", action="store_false")
    parser.add_argument("--idle-vram-mib", type=int, default=None)
    parser.add_argument("--active-vram-mib", type=int, default=None)
    args = parser.parse_args()

    results = [
        query_prompt(args.base_url, pid, prompt, args.max_tokens, args.disable_thinking)
        for pid, prompt in DEFAULT_PROMPTS
    ]

    tps_vals = [r["timings"].get("predicted_per_second", 0.0) for r in results]
    draft_n = sum(int(r["timings"].get("draft_n", 0) or 0) for r in results)
    draft_acc = sum(int(r["timings"].get("draft_n_accepted", 0) or 0) for r in results)

    summary = {
        "avg_decode_tok_per_s": round(sum(tps_vals) / max(1, len(tps_vals)), 2),
        "draft_n": draft_n,
        "draft_n_accepted": draft_acc,
        "draft_acceptance_rate": round(draft_acc / draft_n, 4) if draft_n > 0 else None,
    }
    payload = {
        "idle_vram_mib": args.idle_vram_mib,
        "active_vram_mib": args.active_vram_mib,
        "summary": summary,
        "prompts": results,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
