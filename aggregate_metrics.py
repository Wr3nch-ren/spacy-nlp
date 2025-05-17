import os
import json
import glob

output_dir = "output"
metric_keys = ["uas", "las", "accuracy", "f_score"]  # Add or adjust as needed

fold_metrics = []

for fold_path in sorted(glob.glob(os.path.join(output_dir, "fold*"))):
    metrics_file = os.path.join(fold_path, "metrics.json")
    if not os.path.isfile(metrics_file):
        print(f"metrics.json not found in {fold_path}")
        continue
    with open(metrics_file, "r", encoding="utf8") as f:
        metrics = json.load(f)
    print(f"\nFold: {os.path.basename(fold_path)}")
    fold_result = {"fold": os.path.basename(fold_path)}
    for key in metric_keys:
        value = metrics.get(key)
        if value is not None:
            print(f"  {key}: {value}")
            fold_result[key] = value
    fold_metrics.append(fold_result)

# Aggregate
if fold_metrics:
    print("\n=== Aggregated Results ===")
    for key in metric_keys:
        values = [fm[key] for fm in fold_metrics if key in fm]
        if values:
            avg = sum(values) / len(values)
            print(f"Average {key}: {avg:.4f}")
else:
    print("No metrics found to aggregate.")

input("\nPress Enter to exit...")