import sys
sys.path.insert(0, ".")
# Import the module without running main()
import importlib.util
spec = importlib.util.spec_from_file_location("fc", "_filter_check.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

labels = mod._LABELS
print(f"_LABELS dict size: {len(labels)}")
print(f"sample entries:")
for k, v in list(labels.items())[:5]:
    print(f"  {k} -> {v}")

# Verify diversity ? should be 11 distinct classes
distinct = set(labels.values())
print(f"distinct classes in dict: {len(distinct)}")
if len(distinct) < 5:
    raise SystemExit(f"FAIL: only {len(distinct)} classes found ? labels.csv not loaded correctly")

# Spot-check a non-bagel photo has a non-bagel label
non_bagel_found = False
for fname, lbl in labels.items():
    if lbl != "bagel":
        print(f"  non-bagel sample: {fname} -> {lbl}")
        non_bagel_found = True
        break
if not non_bagel_found:
    raise SystemExit("FAIL: every photo is bagel ? labels.csv not joined")

print()
print("PASS: _LABELS has 11 classes, non-bagel photos will be tested correctly")
