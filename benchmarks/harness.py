"""
Benchmark harness for ActualPlate.

Runs the current pipeline over a dataset with ground-truth labels,
compares predictions to truths, and writes a JSON report.

Usage:

    python -m benchmarks.harness \
        --dataset simplefood45 \
        --endpoint http://127.0.0.1:8080/upload-image \
        --limit 50

Exit codes:
    0   success, results written
    1   no results produced (all requests failed)
    2   invalid arguments or dataset missing
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from benchmarks.metrics import summarize


DEFAULT_DATASET_DIR = Path.home() / "Downloads" / "simple_food_45" / "ordered_dataset"
DEFAULT_LABELS = DEFAULT_DATASET_DIR / "labels.csv"
DEFAULT_IMAGES = DEFAULT_DATASET_DIR
BASELINES_DIR = Path("benchmarks") / "baselines"


def _load_labels(labels_path: Path) -> list[dict]:
    """
    Load labels.csv into a list of dicts with types:
        image: str
        label: str
        weight: float
        volume: float
        energy: float
    """
    rows: list[dict] = []
    with open(labels_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "image": row["image"].strip(),
                    "label": row["label"].strip(),
                    "weight": float(row["weight"]),
                    "volume": float(row["volume"]),
                    "energy": float(row["energy"]),
                }
            )
    return rows


def _post_image(endpoint: str, image_path: Path, timeout: int = 300) -> dict | None:
    """
    POST one image to the backend. Returns the parsed JSON body or None
    if the request failed at the transport level.
    """
    try:
        with open(image_path, "rb") as f:
            response = requests.post(
                endpoint,
                files={"file": (image_path.name, f, "image/jpeg")},
                timeout=timeout,
            )
        if response.status_code != 200:
            return {"_http_status": response.status_code, "_body": response.text[:300]}
        return response.json()
    except Exception as exc:
        return {"_transport_error": str(exc)}


def _extract_predictions(response: dict) -> dict:
    """
    From a /upload-image response, extract:
        mass_g: total predicted mass across all foods
        calories: total predicted energy
        labels: list of predicted food names
        foods_detected: count of foods in the response
        success: whether the response is a valid result (not an error)
    """
    if response is None:
        return {"success": False, "reason": "no_response"}

    if "_transport_error" in response:
        return {"success": False, "reason": "transport", "detail": response["_transport_error"]}

    if "_http_status" in response:
        return {"success": False, "reason": f"http_{response['_http_status']}"}

    if "error" in response:
        return {"success": False, "reason": "pipeline_error", "detail": response["error"]}

    foods = response.get("foods", [])
    if not foods:
        return {"success": False, "reason": "empty_foods"}

    total_mass = sum(f.get("mass_g", 0) or 0 for f in foods)
    total_calories = response.get("total_nutrition", {}).get("calories", 0) or 0
    labels = [f.get("food_name", "") for f in foods]

    return {
        "success": True,
        "mass_g": total_mass,
        "calories": total_calories,
        "labels": labels,
        "foods_detected": len(foods),
    }


def run(dataset: str, endpoint: str, limit: int | None) -> dict:
    """
    Run the benchmark. Returns the report dict.
    """
    if dataset != "simplefood45":
        print(f"Unknown dataset: {dataset}", file=sys.stderr)
        sys.exit(2)

    labels_path = DEFAULT_LABELS
    images_dir = DEFAULT_IMAGES

    if not labels_path.exists():
        print(f"Missing labels file: {labels_path}", file=sys.stderr)
        sys.exit(2)
    if not images_dir.exists():
        print(f"Missing images dir: {images_dir}", file=sys.stderr)
        sys.exit(2)

    rows = _load_labels(labels_path)
    if limit is not None:
        rows = rows[:limit]

    print(f"Loaded {len(rows)} labeled images.")
    print(f"Endpoint: {endpoint}")
    print()

    results = []
    successes = 0
    failures = 0
    failure_reasons: dict[str, int] = {}

    start = time.time()

    for i, row in enumerate(rows, start=1):
        image_path = images_dir / row["image"]
        if not image_path.exists():
            failure_reasons["missing_image"] = failure_reasons.get("missing_image", 0) + 1
            failures += 1
            continue

        response = _post_image(endpoint, image_path)
        pred = _extract_predictions(response)

        record = {
            "image": row["image"],
            "label": row["label"],
            "truth_weight": row["weight"],
            "truth_volume": row["volume"],
            "truth_energy": row["energy"],
        }

        if pred.get("success"):
            record["pred_mass_g"] = pred["mass_g"]
            record["pred_calories"] = pred["calories"]
            record["pred_labels"] = pred["labels"]
            successes += 1
        else:
            record["failure"] = pred.get("reason")
            if pred.get("detail"):
                record["failure_detail"] = pred["detail"]
            reason = pred.get("reason", "unknown")
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            failures += 1

        results.append(record)

        if i % 5 == 0 or i == len(rows):
            elapsed = time.time() - start
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (len(rows) - i) / rate if rate > 0 else 0
            print(
                f"  [{i}/{len(rows)}] success={successes} fail={failures} "
                f"({rate:.2f} img/s, eta {remaining/60:.1f} min)"
            )

    # Compute metrics on successful rows only
    successful_records = [r for r in results if "pred_mass_g" in r]

    mass_metrics = {}
    calorie_metrics = {}
    if successful_records:
        pred_mass = [r["pred_mass_g"] for r in successful_records]
        true_mass = [r["truth_weight"] for r in successful_records]
        pred_cal = [r["pred_calories"] for r in successful_records]
        true_cal = [r["truth_energy"] for r in successful_records]
        mass_metrics = summarize(pred_mass, true_mass)
        calorie_metrics = summarize(pred_cal, true_cal)

    report = {
        "dataset": dataset,
        "endpoint": endpoint,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "total_images": len(rows),
        "successes": successes,
        "failures": failures,
        "failure_reasons": failure_reasons,
        "mass_metrics": mass_metrics,
        "calorie_metrics": calorie_metrics,
        "results": results,
    }

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="ActualPlate benchmark harness")
    parser.add_argument("--dataset", required=True, choices=["simplefood45"])
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8080/upload-image",
        help="Backend /upload-image URL",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only run first N rows")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: benchmarks/baselines/<version>_<dataset>.json)",
    )
    args = parser.parse_args()

    report = run(args.dataset, args.endpoint, args.limit)

    if report["successes"] == 0:
        print()
        print("No successful predictions. Nothing to write.")
        print("Failure reasons:", report["failure_reasons"])
        return 1

    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    output = Path(args.output) if args.output else BASELINES_DIR / f"v3.6.0_{args.dataset}.json"

    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print()
    print(f"Report written to: {output}")
    print()
    print("Mass metrics (g):")
    print(json.dumps(report["mass_metrics"], indent=2))
    print()
    print("Calorie metrics (kcal):")
    print(json.dumps(report["calorie_metrics"], indent=2))
    print()
    print(f"Successes: {report['successes']} / {report['total_images']}")
    print(f"Failures:  {report['failures']}")
    print(f"Failure reasons: {report['failure_reasons']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())