import os
import re
import subprocess
import sys

train_dir = os.path.join("content", "train")
dev_dir = os.path.join("content", "dev")
output_dir = "output"
config_path = "config.cfg"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List all dev files
dev_files = [f for f in os.listdir(dev_dir) if f.endswith(".spacy")]
dev_files.sort()

for dev_file in dev_files:
    print(f"\nProcessing dev file: {dev_file}")
    # Extract fold number using regex (matches fold_1, fold_10, etc.)
    m = re.search(r"fold_(\d+)", dev_file)
    if not m:
        print(f"  Could not extract fold number from dev file: {dev_file}")
        continue
    foldnum = m.group(1)
    print(f"  Extracted fold number: {foldnum}")

    # Find matching train file
    train_files = [f for f in os.listdir(train_dir) if re.search(rf"train_{foldnum}\b", f)]
    if not train_files:
        print(f"  No matching train file for dev fold {foldnum} (searched all train files).")
        continue
    train_file = train_files[0]
    print(f"  Found matching train file: {train_file}")

    train_path = os.path.abspath(os.path.join(train_dir, train_file))
    dev_path = os.path.abspath(os.path.join(dev_dir, dev_file))
    fold_output = os.path.abspath(os.path.join(output_dir, f"fold{foldnum}"))
    log_file = os.path.abspath(os.path.join(output_dir, f"fold{foldnum}.txt"))

    print(f"  Training fold {foldnum} with train: {train_path} and dev: {dev_path}")
    print(f"  Logging output to: {log_file}")
    with open(log_file, "w", encoding="utf-8") as logf:
        process = subprocess.Popen([
            sys.executable, "-m", "spacy", "train", config_path,
            "--output", fold_output,
            "--paths.train", train_path,
            "--paths.dev", dev_path
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True)
        for line in iter(process.stdout.readline, ''):
            print(line, end='')   # Print to terminal
            logf.write(line)      # Write to file
            sys.stdout.flush()    # Ensure immediate output
        process.stdout.close()
        process.wait()
    if process.returncode == 0:
        print(f"  Fold {foldnum} training finished.")
    else:
        print(f"  Training failed for fold {foldnum} (exit code {process.returncode}).")

print("\nAll training folds are complete!")
input("Press Enter to exit...")