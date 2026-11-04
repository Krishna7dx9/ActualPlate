import csv
import os
from pathlib import Path

RESULTS_DIR = Path("benchmarks/ablation/results")

def find_latest(prefix):
    files = list(RESULTS_DIR.glob(f"{prefix}*.csv"))
    if not files:
        return None
    return max(files, key=os.path.getctime)

print("=" * 80)
print(" " * 20 + "ACTUALPLATE ACCURACY TABLE")
print("=" * 80)
print(f"{'Metric':<30} | {'Value':<15} | {'Dataset / Source'}")
print("-" * 80)

# 1. Gemma & End-to-End Volume (SimpleFood45)
sf45_file = find_latest("simplefood45_")
if sf45_file:
    with open(sf45_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if rows:
            # Assuming the CSV has 'label' and 'volume_mape' columns
            mape_key = next((k for k in rows[0].keys() if "mape" in k.lower() or "error" in k.lower()), None)
            if mape_key:
                avg_mape = sum(float(r[mape_key]) for r in rows if r[mape_key]) / len(rows)
                print(f"{'1. Volume MAPE':<30} | {avg_mape:>5.1f}%         | SimpleFood45 (End-to-End)")
            else:
                print(f"{'1. Volume MAPE':<30} | {'No Data':<15} | SimpleFood45")
else:
    print(f"{'1. Volume MAPE':<30} | {'Pending':<15} | SimpleFood45")

# 2. Depth Accuracy (MetaFood3D)
depth_file = find_latest("depth_quality_metafood3d_")
if depth_file:
    with open(depth_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if rows:
            rmse_key = next((k for k in rows[0].keys() if "rmse" in k.lower()), None)
            absrel_key = next((k for k in rows[0].keys() if "absrel" in k.lower() or "rel" in k.lower()), None)
            
            if rmse_key:
                avg_rmse = sum(float(r[rmse_key]) for r in rows if r[rmse_key]) / len(rows)
                print(f"{'2. Depth RMSE':<30} | {avg_rmse:>5.3f} m       | MetaFood3D (Depth Pro)")
            if absrel_key:
                avg_absrel = sum(float(r[absrel_key]) for r in rows if r[absrel_key]) / len(rows)
                print(f"{'3. Depth AbsRel':<30} | {avg_absrel:>5.3f}         | MetaFood3D (Depth Pro)")
else:
    print(f"{'2. Depth RMSE':<30} | {'Pending':<15} | MetaFood3D")

# 3. Formula Accuracy (MetaFood3D)
formula_file = find_latest("formula_diagnostic_metafood3d_")
if formula_file:
    with open(formula_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if rows:
            # Look for the v5 formula error column
            v5_key = next((k for k in rows[0].keys() if "v5" in k.lower() or "volume" in k.lower()), None)
            if v5_key:
                avg_v5_err = sum(float(r[v5_key]) for r in rows if r[v5_key]) / len(rows)
                print(f"{'4. Formula Math Error':<30} | {avg_v5_err:>5.1f}%         | MetaFood3D (v5 Math)")
else:
    print(f"{'4. Formula Math Error':<30} | {'Pending':<15} | MetaFood3D")

# 4. Bbox and Segmentation Accuracy
print("-" * 80)
print(f"{'5. Bbox mAP@50':<30} | {'Pending':<15} | Run: python -m benchmarks.ablation.mesh_projector_audit")
print(f"{'6. Mask mIoU':<30} | {'Pending':<15} | Run: python -m benchmarks.ablation.mask_overlay")
print("=" * 80)
print("NOTE: You have mf3d_mesh and mf3d_partial downloaded.")
print("To generate Bbox and Mask accuracy, we must run the mesh projection scripts.")
