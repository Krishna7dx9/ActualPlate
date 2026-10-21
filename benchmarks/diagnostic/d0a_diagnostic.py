"""D0a: characterize volume error. CSV only. No GPU, no server.

Answers:
  - Is the error tight or wide within a unique item?
  - Does log r correlate with hcap, or is it per-photo noise?
  - How many unique items per class? (>=5 needed for any held-out gate)

Does NOT answer isotropic vs height-only. With a per-class constant,
log r = 3 log s and log r = log k are the same model (k = s^3).
That split moves to P2 (ArUco PnP + manual measurement).
"""
import numpy as np
import pandas as pd
from pathlib import Path

PRED = Path("benchmarks/ablation/results/filter_check_multiclass.csv")
LABELS = Path.home() / "Downloads" / "simple_food_45" / "ordered_dataset" / "labels.csv"
OUT = Path("benchmarks/diagnostic/d0a_report.csv")
MIN_ITEMS = 5

# --- Load ---
raw = pd.read_csv(PRED)
print("=" * 78)
print("COUNT RECONCILIATION")
print("=" * 78)
print(f"Rows in {PRED.name}:        {len(raw)}")
raw["vol_cm3"] = pd.to_numeric(raw["vol_cm3"], errors="coerce")
n_vol = int((raw["vol_cm3"] > 0).sum())
print(f"Rows with vol_cm3 > 0:          {n_vol}")
print(f"Rows with vol_cm3 null/err:     {len(raw) - n_vol}")

lab = pd.read_csv(LABELS)
print(f"Rows in labels.csv:             {len(lab)}")

# --- Join on filename ---
pred = raw.rename(columns={"file": "photo"})
lab = lab.rename(columns={"image": "photo"})
df = pred.merge(lab, on="photo", how="inner", suffixes=("", "_lab"))
print(f"Rows after join on photo:       {len(df)}")

df = df[(df["vol_cm3"] > 0) & (df["volume"] > 0)].copy()
print(f"Usable (both volumes > 0):      {len(df)}")

# --- Item cluster ---
df["item"] = (
    df["label"].astype(str) + "|" +
    df["weight"].astype(str) + "|" +
    df["volume"].round(2).astype(str)
)

# --- r = V_true / V_pred ---
df["r"] = df["volume"] / df["vol_cm3"]
df["logr"] = np.log(df["r"])

# --- hcap availability ---
if "hcap_mm" in df.columns:
    df["hcap_mm"] = pd.to_numeric(df["hcap_mm"], errors="coerce")
    n_hcap = int(df["hcap_mm"].notna().sum())
else:
    n_hcap = 0
print(f"Rows with hcap_mm available:    {n_hcap}")

# --- Per class ---
rows = []
for cls, g in df.groupby("label"):
    items = g.groupby("item")["logr"]
    n_items = int(items.ngroups)
    r = g["r"]

    base_ape = (np.abs(1.0 / r - 1.0) * 100.0).median()

    within_var = items.var().mean(skipna=True)
    within_sd = float(np.sqrt(within_var)) if np.isfinite(within_var) else np.nan

    item_med = items.median()
    between_sd = float(item_med.std()) if n_items > 1 else np.nan

    loo_ape = np.nan
    if n_items >= MIN_ITEMS:
        errs = []
        for it in item_med.index:
            c = float(item_med.drop(it).median())
            lr = g.loc[g["item"] == it, "logr"]
            errs.extend((np.abs(np.exp(c - lr) - 1.0) * 100.0).tolist())
        loo_ape = float(np.median(errs))

    rho_hcap = np.nan
    if n_hcap >= 10 and "hcap_mm" in g.columns:
        v = g[(g["hcap_mm"] > 0)].copy()
        if len(v) >= 10 and v["hcap_mm"].nunique() > 2:
            rho_hcap = float(v["logr"].corr(np.log(v["hcap_mm"]), method="spearman"))

    rows.append({
        "class": cls,
        "photos": len(g),
        "items": n_items,
        "r_median": round(float(r.median()), 3),
        "r_iqr": round(float(r.quantile(0.75) - r.quantile(0.25)), 3),
        "within_sd": round(within_sd, 3) if np.isfinite(within_sd) else "n/a",
        "between_sd": round(between_sd, 3) if np.isfinite(between_sd) else "n/a",
        "base_ape_pct": round(base_ape, 1),
        "loo_ape_pct": round(loo_ape, 1) if np.isfinite(loo_ape) else "insufficient",
        "rho_hcap": round(rho_hcap, 2) if np.isfinite(rho_hcap) else "n/a",
    })

out = pd.DataFrame(rows).sort_values("class")
print()
print("=" * 78)
print("PER-CLASS CHARACTERIZATION")
print("=" * 78)
print(out.to_string(index=False))
OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False)

# --- Decision ---
print()
print("=" * 78)
print("DECISION")
print("=" * 78)

classes_with_enough = int((out["items"] >= MIN_ITEMS).sum())
total_classes = len(out)
print(f"Classes with >= {MIN_ITEMS} items: {classes_with_enough}/{total_classes}")

within_vals = [r["within_sd"] for r in rows if isinstance(r["within_sd"], (int, float))]
between_vals = [r["between_sd"] for r in rows if isinstance(r["between_sd"], (int, float))]

if classes_with_enough < max(1, total_classes // 2):
    print()
    print("OUTPUT: INSUFFICIENT ITEMS")
    print("Most classes have fewer than 5 unique items. Cannot pick a P3 branch from this data.")
    print("NEXT: P2 (ArUco PnP + manual caliper measurement of 5-10 items)")
elif within_vals and between_vals:
    mw = float(np.mean(within_vals))
    mb = float(np.mean(between_vals))
    print()
    print(f"Mean within-item SD of log r:  {mw:.3f}")
    print(f"Mean between-item SD of log r: {mb:.3f}")
    print(f"Ratio between/within:          {mb/mw if mw > 0 else float('inf'):.2f}")
    print()
    if mb > 2.0 * mw:
        print("OUTPUT: TIGHT WITHIN ITEM, WIDE BETWEEN ITEMS")
        print("Per-item systematic error. Fixable by per-item correction or scale cue.")
        print("NEXT: P2 (depth A/B + decomposition) -> P3-B (scale fusion)")
    elif mw > mb:
        print("OUTPUT: WIDE WITHIN ITEM")
        print("Per-photo noise dominates. Shape / mask / plane errors.")
        print("NEXT: P3-C (shape fix) first, then P2")
    else:
        print("OUTPUT: MIXED")
        print("Neither tight nor wide dominates. P2 decomposition is required to split height vs footprint.")
        print("NEXT: P2 first (ArUco PnP + manual measurement)")
else:
    print()
    print("OUTPUT: INSUFFICIENT DATA")
    print("Not enough non-null values to compute within/between spread.")
    print("NEXT: inspect the CSV; confirm both volume columns parsed.")

print()
print(f"Report saved: {OUT}")
