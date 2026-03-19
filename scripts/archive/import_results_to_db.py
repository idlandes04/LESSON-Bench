#!/usr/bin/env python3
"""Import existing JSON results into the SQLite results store.

Reads all *_sb2_all.json and *_complete.json files from results directories,
plus the sb2_results_db.json for status flags, and populates lesson_bench.db.

Usage:
    python scripts/import_results_to_db.py
    python scripts/import_results_to_db.py --db results/lesson_bench.db
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lesson.results.store import ResultsStore


CORE_CONDITIONS = ["correction", "practice_only", "error_only", "no_feedback"]


def import_sb2_all_files(store: ResultsStore, results_root: Path) -> int:
    """Import *_sb2_all.json files from all result directories."""
    imported = 0

    for results_dir in sorted(results_root.iterdir()):
        if not results_dir.is_dir():
            continue

        # Look for run_config.json to get run metadata
        config_path = results_dir / "run_config.json"
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

        run_id = results_dir.name
        version = config.get("version", "")
        conditions = config.get("conditions", CORE_CONDITIONS)

        # Ensure run exists
        store.save_run(
            run_id=run_id,
            version=version,
            config=config,
            results_dir=str(results_dir),
        )

        # Import *_sb2_all.json files
        for json_file in sorted(results_dir.glob("*_sb2_all.json")):
            try:
                with open(json_file) as f:
                    model_data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"  SKIP {json_file.name}: {e}")
                continue

            model = model_data.get("model", json_file.stem.replace("_sb2_all", ""))
            provider = model_data.get("provider", "unknown")

            store.save_model_results(
                run_id=run_id,
                model=model,
                provider=provider,
                model_data=model_data,
                conditions=conditions,
            )
            imported += 1
            print(f"  Imported {json_file.name} -> {model} ({len(conditions)} conditions)")

        # Import *_complete.json files (from free_models runs)
        for json_file in sorted(results_dir.glob("*_complete.json")):
            try:
                with open(json_file) as f:
                    model_data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"  SKIP {json_file.name}: {e}")
                continue

            model = model_data.get("model", json_file.stem.replace("_complete", ""))
            provider = model_data.get("provider", "unknown")

            # These files may have a different structure — check for sb2_ keys
            found_conditions = [
                k.replace("sb2_", "")
                for k in model_data
                if k.startswith("sb2_") and not k.endswith("_elapsed_s") and not k.endswith("_error")
            ]
            if found_conditions:
                store.save_model_results(
                    run_id=run_id,
                    model=model,
                    provider=provider,
                    model_data=model_data,
                    conditions=found_conditions,
                )
                imported += 1
                print(f"  Imported {json_file.name} -> {model} ({len(found_conditions)} conditions)")

    return imported


def import_sb2_results_db(store: ResultsStore, db_path: Path) -> int:
    """Import status flags from the JSON results DB."""
    if not db_path.exists():
        return 0

    with open(db_path) as f:
        db = json.load(f)

    run_id = "sb2_pilot_resume"
    config = db.get("config", {})
    results_dir = config.get("results_dir", "")

    store.save_run(
        run_id=run_id,
        version=config.get("version", ""),
        config=config,
        results_dir=results_dir,
    )

    imported = 0
    for model_name, model_data in db.get("models", {}).items():
        provider = model_data.get("provider", "unknown")
        for condition, cell_data in model_data.get("cells", {}).items():
            store.save_cell(
                run_id=run_id,
                model=model_name,
                provider=provider,
                condition=condition,
                cell_data=cell_data,
            )
            imported += 1

    print(f"  Imported {imported} cells from {db_path.name}")
    return imported


def main():
    parser = argparse.ArgumentParser(description="Import results into SQLite store")
    parser.add_argument("--db", type=str, default=None, help="SQLite DB path")
    args = parser.parse_args()

    results_root = Path("results")
    if not results_root.exists():
        print("ERROR: results/ directory not found")
        sys.exit(1)

    db_path = Path(args.db) if args.db else None
    store = ResultsStore(db_path) if db_path else ResultsStore()

    print(f"Importing results into {store._db_path}")
    print()

    # Import JSON result files
    print("--- Importing *_sb2_all.json and *_complete.json files ---")
    n_files = import_sb2_all_files(store, results_root)

    # Import JSON results DB
    print("\n--- Importing sb2_results_db.json ---")
    json_db = results_root / "sb2_results_db.json"
    n_db = import_sb2_results_db(store, json_db)

    store.close()

    print(f"\nDone: {n_files} result files + {n_db} DB cells imported")
    print(f"Database: {store._db_path}")


if __name__ == "__main__":
    main()
