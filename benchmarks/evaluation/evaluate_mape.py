import csv
from collections import defaultdict

def calculate_mape(csv_path):
    """
    Reads the benchmark CSV and calculates the MAPE broken down by food category.
    """
    category_errors = defaultdict(list)
    total_rows = 0
    skipped_rows = 0
    
    print(f"Reading data from: {csv_path}\n")
    
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            total_rows += 1
            
            # Skip rows without a prediction or ground truth
            if not row.get('vol_A') or not row.get('gt_volume_cm3'):
                skipped_rows += 1
                continue
                
            try:
                gt = float(row['gt_volume_cm3'])
                pred = float(row['vol_A'])
            except ValueError:
                skipped_rows += 1
                continue

            if gt <= 0:
                skipped_rows += 1
                continue

            # Calculate Absolute Percentage Error
            error = abs(pred - gt) / gt
            category = row['category']
            
            category_errors[category].append(error)

    print(f"{'Category':<20} | {'Images':>6} | {'MAPE %':>10}")
    print("-" * 42)
    
    total_error_sum = 0
    total_evaluated = 0

    # Calculate MAPE per category
    for cat, errors in sorted(category_errors.items()):
        cat_mape = (sum(errors) / len(errors)) * 100
        print(f"{cat:<20} | {len(errors):>6} | {cat_mape:>9.1f}%")
        
        total_error_sum += sum(errors)
        total_evaluated += len(errors)
        
    print("-" * 42)
    
    # Calculate overall MAPE
    if total_evaluated > 0:
        overall_mape = (total_error_sum / total_evaluated) * 100
        print(f"{'OVERALL':<20} | {total_evaluated:>6} | {overall_mape:>9.1f}%")
    else:
        print("No valid rows found for evaluation.")

    print("\n--- DATA COVERAGE WARNING ---")
    print(f"Total Rows in CSV: {total_rows}")
    print(f"Successfully Evaluated: {total_evaluated}")
    if skipped_rows > 0:
        print(f"⚠️ WARNING: {skipped_rows} rows were skipped because they are missing predictions (vol_A is empty).")
        print("This usually means the benchmark run timed out or crashed before finishing.")

if __name__ == "__main__":
    CSV_FILE = "benchmarks/ablation/results/simplefood45_20261016T100545Z.csv"
    calculate_mape(CSV_FILE)