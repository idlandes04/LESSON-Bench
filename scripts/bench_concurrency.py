#!/usr/bin/env python3
"""Benchmark LM Studio parallel inference at different concurrency levels.

Sends N identical STS-style prompts and measures wall-clock time + throughput
at concurrency levels 1, 2, 4, 6, 8, 10, 12, 16.

Usage:
    python scripts/bench_concurrency.py
    python scripts/bench_concurrency.py --n-prompts 32 --levels 1,4,8,16
"""

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LMSTUDIO_PORT = 1234
MODEL_ID = "qwen3.5-27b"

# A realistic STS prompt (similar length to what the eval pipeline sends)
PROMPT = (
    "Below are 4 examples of a symbolic transformation system.\n"
    "The system uses these symbols: ◈ ⬡ ⟐ ⧫ △ ▽\n"
    "Study the pattern, then predict the output for the final input.\n"
    "\n"
    "Input: ◈ ⬡ ⟐ ⧫ △  →  Output: ⬡ ⟐ ⧫ △ ◈\n"
    "Input: ⧫ △ ▽ ◈ ⬡  →  Output: △ ▽ ◈ ⬡ ⧫\n"
    "Input: ⟐ ⧫ △ ▽ ◈  →  Output: ⧫ △ ▽ ◈ ⟐\n"
    "Input: ▽ ◈ ⬡ ⟐ ⧫  →  Output: ◈ ⬡ ⟐ ⧫ ▽\n"
    "\n"
    "Input: △ ▽ ◈ ⬡ ⟐\n"
    'Respond with ONLY a JSON object: {"output": "YOUR_ANSWER"}\n'
    "Use only the symbols listed above in your answer."
)

JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "sts_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"output": {"type": "string"}},
            "required": ["output"],
            "additionalProperties": False,
        },
    },
}


def send_one(client: OpenAI, prompt_text: str) -> dict:
    """Send a single prompt and return timing + response info."""
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": prompt_text},
        ],
        max_tokens=64,
        temperature=0.0,
        response_format=JSON_SCHEMA,
    )
    elapsed = time.perf_counter() - t0
    content = resp.choices[0].message.content or ""
    usage = resp.usage
    return {
        "elapsed": elapsed,
        "content": content,
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
    }


def bench_level(client: OpenAI, n_prompts: int, concurrency: int) -> dict:
    """Run n_prompts at given concurrency level and return stats."""
    results = []
    t_wall_start = time.perf_counter()

    if concurrency == 1:
        for _ in range(n_prompts):
            results.append(send_one(client, PROMPT))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(send_one, client, PROMPT) for _ in range(n_prompts)]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())

    t_wall = time.perf_counter() - t_wall_start

    per_request = [r["elapsed"] for r in results]
    total_completion_tokens = sum(r["completion_tokens"] for r in results)
    total_prompt_tokens = sum(r["prompt_tokens"] for r in results)

    return {
        "concurrency": concurrency,
        "n_prompts": n_prompts,
        "wall_time": round(t_wall, 2),
        "throughput_req_per_sec": round(n_prompts / t_wall, 2),
        "avg_latency": round(sum(per_request) / len(per_request), 3),
        "min_latency": round(min(per_request), 3),
        "max_latency": round(max(per_request), 3),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "tokens_per_sec": round(total_completion_tokens / t_wall, 1) if t_wall > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark LM Studio concurrency")
    parser.add_argument("--n-prompts", type=int, default=24,
                        help="Number of prompts per concurrency level (default: 24)")
    parser.add_argument("--levels", default="1,2,4,6,8,10,12,16",
                        help="Comma-separated concurrency levels to test")
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    n = args.n_prompts

    client = OpenAI(base_url=f"http://localhost:{LMSTUDIO_PORT}/v1", api_key="lm-studio")

    # Warmup: 2 requests to ensure model is loaded and KV cache is warm
    print("Warming up...")
    for _ in range(2):
        send_one(client, PROMPT)
    print("Warmup done.\n")

    print(f"{'Concurrency':>12} {'Wall (s)':>10} {'Req/s':>8} {'Avg Lat':>9} "
          f"{'Min Lat':>9} {'Max Lat':>9} {'Tok/s':>8}")
    print("-" * 75)

    all_results = []
    for level in levels:
        result = bench_level(client, n, level)
        all_results.append(result)
        print(f"{result['concurrency']:>12} {result['wall_time']:>10.2f} "
              f"{result['throughput_req_per_sec']:>8.2f} "
              f"{result['avg_latency']:>9.3f} {result['min_latency']:>9.3f} "
              f"{result['max_latency']:>9.3f} {result['tokens_per_sec']:>8.1f}")

    # Find optimal
    best = max(all_results, key=lambda r: r["throughput_req_per_sec"])
    print(f"\nOptimal concurrency: {best['concurrency']} "
          f"({best['throughput_req_per_sec']} req/s, "
          f"{best['tokens_per_sec']} tok/s)")

    # Save raw results
    out_path = Path("results") / "concurrency_bench.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved: {out_path}")


if __name__ == "__main__":
    main()
