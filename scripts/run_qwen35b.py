#!/usr/bin/env python3
"""Run experiments for Qwen3.5-35B-A3B nothink only."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from lesson.models.registry import get_local_client
from lesson.eval.pilot import run_sb1_pilot
from lesson.eval.sb2_pilot import run_sb2_pilot


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results") / f"qwen35b_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    client = get_local_client("qwen3.5-35b-a3b-nothink")
    model_results = {"model": "qwen3.5-35b-a3b-nothink", "type": "local"}

    # SB1
    print("\n--- SB1 ---")
    t0 = time.time()
    try:
        sb1 = run_sb1_pilot(client=client, tiers=[1, 2, 3], n_values=[4, 8], n_instances=3)
        model_results["sb1"] = sb1
        print(f"\n  SB1 done in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"  SB1 FAILED: {e}")
        import traceback; traceback.print_exc()

    with open(results_dir / "qwen35b_sb1.json", "w") as f:
        json.dump(model_results, f, indent=2, default=str)

    # SB2 correction + practice_only
    print("\n--- SB2 correction + practice_only ---")
    t0 = time.time()
    try:
        sb2 = run_sb2_pilot(client=client, tier=2, n_initial_examples=8,
                           n_instances=3, n_turns=8, conditions=["correction", "practice_only"])
        model_results["sb2_main"] = sb2
        print(f"\n  SB2 main done in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"  SB2 FAILED: {e}")
        import traceback; traceback.print_exc()

    with open(results_dir / "qwen35b_sb2.json", "w") as f:
        json.dump(model_results, f, indent=2, default=str)

    # SB2 error_only
    print("\n--- SB2 error_only ---")
    t0 = time.time()
    try:
        sb2_err = run_sb2_pilot(client=client, tier=2, n_initial_examples=8,
                               n_instances=3, n_turns=8, conditions=["error_only"])
        model_results["sb2_error_only"] = sb2_err
        print(f"\n  SB2 error_only done in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"  SB2 FAILED: {e}")
        import traceback; traceback.print_exc()

    with open(results_dir / "qwen35b_all.json", "w") as f:
        json.dump(model_results, f, indent=2, default=str)

    print(f"\nAll results: {results_dir}")


if __name__ == "__main__":
    main()
