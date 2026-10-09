"""
Diagnose one image: print every intermediate value the pipeline
produces and compare against ground truth.

Usage:

    python -m benchmarks.diagnose --image 20230927_102352.jpg

Requires the backend and Colab to be running.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
from pathlib import Path

import requests


DATASET = Path.home() / "Downloads" / "simple_food_45" / "ordered_dataset"
LABELS = DATASET / "labels.csv"
BACKEND = "http://127.0.0.1:8080"


def load_truth(image_name: str) -> dict | None:
    with open(LABELS, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["image"] == image_name:
                return {
                    "label": row["label"],
                    "weight_g": float(row["weight"]),
                    "volume_ml": float(row["volume"]),
                    "energy_kcal": float(row["energy"]),
                }
    return None


def stage_1_gemma(image_bytes: bytes) -> dict:
    """
    Call Gemma directly (bypassing the backend) to see what it returns.
    """
    from vision_service.recognition.food_recognition import detect_food

    print("=" * 70)
    print("STAGE 1 — GEMMA (recognition + density)")
    print("=" * 70)
    result = detect_food(image_bytes)
    print(json.dumps(result, indent=2))
    return result


def stage_2_colab(image_path: Path, labels: list[str]) -> dict:
    """
    Call Colab /detect directly to see what DINO + SAM2 + Depth Pro return.
    """
    print()
    print("=" * 70)
    print("STAGE 2 — COLAB (detection + volume)")
    print("=" * 70)
    print("Labels sent:", labels)

    import os
    from dotenv import load_dotenv
    load_dotenv()
    endpoint = os.getenv("VISION_INFERENCE_ENDPOINT")
    if not endpoint:
        print("VISION_INFERENCE_ENDPOINT not set in .env")
        sys.exit(2)

    with open(image_path, "rb") as f:
        response = requests.post(
            f"{endpoint}/detect",
            files={"file": (image_path.name, f, "image/jpeg")},
            data={"labels": ", ".join(labels)},
            timeout=300,
        )
    response.raise_for_status()
    result = response.json()
    print(json.dumps(result, indent=2))
    return result


def stage_3_backend(image_path: Path) -> dict:
    """
    Call the full backend /upload-image to see the final response.
    """
    print()
    print("=" * 70)
    print("STAGE 3 — BACKEND (full pipeline)")
    print("=" * 70)
    with open(image_path, "rb") as f:
        response = requests.post(
            f"{BACKEND}/upload-image",
            files={"file": (image_path.name, f, "image/jpeg")},
            timeout=300,
        )
    response.raise_for_status()
    result = response.json()
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Image filename in the dataset")
    args = parser.parse_args()

    image_path = DATASET / args.image
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return 2

    truth = load_truth(args.image)
    if truth is None:
        print(f"No ground truth for {args.image}")
        return 2

    print()
    print("=" * 70)
    print("GROUND TRUTH")
    print("=" * 70)
    print(json.dumps(truth, indent=2))

    image_bytes = image_path.read_bytes()

    gemma = stage_1_gemma(image_bytes)
    labels = [item["search_query"] for item in gemma]

    colab = stage_2_colab(image_path, labels)

    backend = stage_3_backend(image_path)

    print()
    print("=" * 70)
    print("STAGE-BY-STAGE COMPARISON")
    print("=" * 70)

    # Gemma stage
    print()
    print("Gemma: identified %d food(s)" % len(gemma))
    for item in gemma:
        print(f"  - {item['search_query']} (density {item['density_g_cm3']} g/cm³)")

    # Colab stage
    foods = colab.get("foods", [])
    print()
    print(f"Colab: returned {len(foods)} detection(s)")
    total_volume_cm3 = 0.0
    for f in foods:
        vol = f.get("volume_cm3")
        print(f"  - {f.get('label')} vol={vol} cm³ status={f.get('volume_status')}")
        if vol:
            total_volume_cm3 += vol
    print(f"Colab total volume: {total_volume_cm3:.2f} cm³")
    print(f"Ground-truth volume: {truth['volume_ml']:.2f} mL")
    if truth["volume_ml"] > 0:
        ratio = total_volume_cm3 / truth["volume_ml"]
        print(f"VOLUME RATIO (predicted / truth): {ratio:.2f}x")

    # Backend stage
    print()
    foods_final = backend.get("foods", [])
    total_mass = sum(f.get("mass_g", 0) or 0 for f in foods_final)
    total_cal = backend.get("total_nutrition", {}).get("calories", 0)
    print(f"Backend: {len(foods_final)} food(s) in response")
    print(f"Total mass: {total_mass:.2f} g (ground truth: {truth['weight_g']:.2f} g)")
    print(f"Total calories: {total_cal:.2f} kcal (ground truth: {truth['energy_kcal']:.2f} kcal)")
    if truth["weight_g"] > 0:
        print(f"MASS RATIO: {total_mass / truth['weight_g']:.2f}x")
    if truth["energy_kcal"] > 0:
        print(f"CALORIE RATIO: {total_cal / truth['energy_kcal']:.2f}x")

    # Verdict
    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    if truth["volume_ml"] > 0 and total_volume_cm3 > 0:
        vol_ratio = total_volume_cm3 / truth["volume_ml"]
        if vol_ratio > 3:
            print(f"  Volume is {vol_ratio:.1f}x too high → volume engine is the flaw.")
        elif vol_ratio < 0.33:
            print(f"  Volume is {vol_ratio:.1f}x too low → volume engine under-counts.")
        else:
            print(f"  Volume is within 3x of truth — volume engine is acceptable.")

    return 0


if __name__ == "__main__":
    sys.exit(main())