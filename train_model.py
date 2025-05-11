import os
import sys
import json
import shutil
import tempfile
import time
import re
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

class TrainingMonitor:
    """Monitor training progress and collect metrics from spaCy's output."""
    
    def __init__(self):
        self.metrics = []
        self.output_lines = []
        self.current_epoch = 0
        self.completed = False
        self.error = None
    
def process_line(self, line: str) -> None:
        """Process a line of output from the training process."""
        self.output_lines.append(line)
        
        # Extract epoch information
        if "E    #" in line and "LOSS PARSER" in line:
            # This is the header line, skip
            pass
        elif re.match(r"^\s*\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s*$", line):
            # This matches the metrics line in the output
            try:
                parts = line.strip().split()
                if len(parts) >= 6:
                    epoch = int(parts[0])
                    self.current_epoch = epoch
                    score = float(parts[4])
                    las = float(parts[4])  # LAS is the same as score in this context
                    uas = float(parts[5])
                    
                    self.metrics.append({
                        "epoch": epoch,
                        "score": score,
                        "las": las,
                        "uas": uas
                    })
                    print(f"Epoch {epoch} - Score: {score:.2f}, LAS: {las:.2f}, UAS: {uas:.2f}")
            except Exception as e:
                # Skip lines that don't match the expected format
                pass
        
        # Check for training completion
        if "Saving model" in line:
            self.completed = True
        
        # Check for errors
        if "Error" in line or "ERROR" in line or "Exception" in line:
            self.error = line
            
    def get_last_metrics(self) -> Optional[Dict[str, float]]:
        """Get the metrics from the last epoch."""
        if not self.metrics:
            return None
        return self.metrics[-1]
    
    def get_full_output(self) -> str:
        """Get the full output as a string."""
        return "\n".join(self.output_lines)

def load_config(config_path: str = "training_config.json") -> Dict[str, Any]:
    """Load the training configuration from a JSON file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            print(f"✅ Loaded configuration from {config_path}")
            return config
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        print("Using default configuration...")
        return {
            "language": "th",
            "pipeline": ["parser"],
            "batch_size": 1000,
            "max_epochs": 10,
            "dropout": 0.2,
            "patience": 3,
            "eval_frequency": 1000,
            "seed": 42,
            "gpu_allocator": "cpu",
            "learn_rate": 0.001,
            "use_gpu": False,
            "show_progress": True
        }

def validate_path(path: str, is_dir: bool = True, create_if_missing: bool = False) -> bool:
    """Validate that a path exists and is accessible."""
    # Convert to absolute path for better reliability
    abs_path = os.path.abspath(path)
    path_obj = Path(abs_path)
    
    if is_dir:
        if os.path.isdir(abs_path):
            print(f"✅ Directory exists: {abs_path}")
            return True
        elif create_if_missing:
            try:
                os.makedirs(abs_path, exist_ok=True)
                print(f"✅ Created directory: {abs_path}")
                return True
            except Exception as e:
                print(f"❌ Failed to create directory '{abs_path}': {e}")
                return False
        else:
            print(f"❌ Directory not found: {abs_path}")
            return False
    else:
        if os.path.isfile(abs_path):
            print(f"✅ File exists: {abs_path}")
            return True
        else:
            print(f"❌ File not found: {abs_path}")
            return False

def get_training_files(train_folder: str, dev_folder: str) -> Tuple[List[str], List[str]]:
    """Get sorted lists of training and development files."""
    try:
        # Ensure absolute paths
        train_folder = os.path.abspath(train_folder)
        dev_folder = os.path.abspath(dev_folder)
        
        print(f"Looking for training files in: {train_folder}")
        print(f"Looking for development files in: {dev_folder}")
        
        train_files = sorted([f for f in os.listdir(train_folder) if f.endswith(".spacy")])
        dev_files = sorted([f for f in os.listdir(dev_folder) if f.endswith(".spacy")])
        
        print(f"Found {len(train_files)} training files and {len(dev_files)} development files")
        
        if len(train_files) != len(dev_files):
            print(f"⚠️ Warning: Mismatch between train ({len(train_files)}) and dev ({len(dev_files)}) files count")
            
        if not train_files:
            print("❌ No training files found")
        if not dev_files:
            print("❌ No development files found")
            
        return train_files, dev_files
    except Exception as e:
        print(f"❌ Error getting training files: {e}")
        print(f"  Error details: {str(e)}")
        return [], []

# Base configuration template with CPU settings for Thai parsing
config_template = """
[paths]
train = "{train_path}"
dev = "{dev_path}"
vectors = null

[system]
gpu_allocator = "{gpu_allocator}"
seed = {seed}

[nlp]
lang = "th"
pipeline = ["parser"]
batch_size = {batch_size}
tokenizer = {{"@tokenizers": "spacy.th.ThaiTokenizer"}}

[components]

[components.parser]
factory = "parser"

[components.parser.model]
@architectures = "spacy.TransitionBasedParser.v2"
state_type = "parser"
extra_state_tokens = false
hidden_width = 64
maxout_pieces = 2
use_upper = false
nO = null

[components.parser.model.tok2vec]
@architectures = "spacy.Tok2Vec.v2"

[components.parser.model.tok2vec.embed]
@architectures = "spacy.MultiHashEmbed.v2"
width = 96
attrs = ["NORM", "PREFIX", "SUFFIX", "SHAPE"]
rows = [5000, 1000, 2500, 2500]
include_static_vectors = false

[components.parser.model.tok2vec.encode]
@architectures = "spacy.MaxoutWindowEncoder.v2"
width = 96
depth = 4
window_size = 1
maxout_pieces = 2

[corpora]

[corpora.train]
@readers = "spacy.Corpus.v1"
path = ${{paths.train}}
max_length = 0
gold_preproc = false

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = ${{paths.dev}}
max_length = 0
gold_preproc = false

[training]
seed = {seed}
gpu_allocator = "{gpu_allocator}"
dropout = {dropout}
max_epochs = {max_epochs}
patience = {patience}
eval_frequency = {eval_frequency}

[training.optimizer]
@optimizers = "Adam.v1"
learn_rate = {learn_rate}

[initialize]
vectors = null
"""

def train_fold(fold_number: int, train_path: str, dev_path: str, output_dir: str, config: Dict[str, Any]) -> Tuple[bool, Optional[TrainingMonitor]]:
    """Train a spaCy model for a fold with improved progress tracking and error handling."""
    # Ensure absolute paths
    train_path = os.path.abspath(train_path)
    dev_path = os.path.abspath(dev_path)
    output_dir = os.path.abspath(output_dir)
    
    config_filename = None
    temp_dir = None
    try:
        # Create the config string for the current fold
        # Normalize paths by replacing backslashes with forward slashes
        normalized_train_path = str(train_path).replace("\\", "/")
        normalized_dev_path = str(dev_path).replace("\\", "/")
        
        # Update parameters with paths
        params = config.copy()
        params.update({
            "train_path": normalized_train_path,
            "dev_path": normalized_dev_path
        })
        
        # Create the config string with all parameters
        config_string = config_template.format(**params)
        
        # Write the config file to a temporary location
        temp_dir = tempfile.mkdtemp()
        config_filename = os.path.join(temp_dir, f"config_fold{fold_number}.cfg")
        
        with open(config_filename, "w", encoding="utf-8") as f:
            f.write(config_string)
        
        print(f"Created configuration file: {config_filename}")
        
        # Define the output directory for the current fold
        os.makedirs(output_dir, exist_ok=True)
        
        # Build the command
        command = ["python", "-m", "spacy", "train", config_filename, "--output", output_dir]
        
        print(f"Running training command: {' '.join(command)}")
        
        # Initialize monitor
        monitor = TrainingMonitor()
        
        # Start the process with pipe for output
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Process the output in real-time
        while True:
            # Read output line by line
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
                
            if line:
                line = line.strip()
                print(f"    {line}")  # Print the line to console with indentation
                monitor.process_line(line)
                
                # Break if there was an error
                if monitor.error:
                    print(f"❌ Error detected: {monitor.error}")
                    break
                
        # Wait for process to complete and get return code
        return_code = process.wait()
        
        # Check if training was successful
        if return_code != 0:
            print(f"❌ Training failed for fold {fold_number} with return code {return_code}")
            return False, monitor
        
        print(f"✅ Training completed successfully for fold {fold_number}")
        
        # Save the metrics to a JSON file
        metrics_file = os.path.join(output_dir, "metrics.json")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump({"metrics": monitor.metrics}, f, indent=2)
        
        return True, monitor
        
    except Exception as e:
        print(f"❌ Error in training fold {fold_number}: {e}")
        return False, None
        
    finally:
        # Clean up the temporary config file and directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                print(f"Warning: Could not remove temporary directory: {e}")
def main():
    """Main function to manage the training process."""
    print(f"SpaCy NLP Model Training Script")
    print(f"===============================")
    
    # Load configuration
    config = load_config()
    
    # Define paths - use absolute paths for better reliability
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_folder = os.path.join(current_dir, "content", "train")
    dev_folder = os.path.join(current_dir, "content", "dev")
    output_base_dir = os.path.join(current_dir, "output")
    
    # Validate paths
    train_valid = validate_path(train_folder, is_dir=True, create_if_missing=False)
    dev_valid = validate_path(dev_folder, is_dir=True, create_if_missing=False)
    output_valid = validate_path(output_base_dir, is_dir=True, create_if_missing=True)
    
    if not (train_valid and dev_valid):
        print("❌ Path validation failed. Please check the directory structure.")
        print(f"Make sure the following directories exist:")
        print(f"  - Training folder: {train_folder}")
        print(f"  - Development folder: {dev_folder}")
        sys.exit(1)
    
    # Get training files
    train_files, dev_files = get_training_files(train_folder, dev_folder)
    
    if not train_files or not dev_files:
        print("❌ No training or development files found.")
        print("Please ensure your .spacy files are in the correct directories:")
        print(f"  - Training files should be in: {train_folder}")
        print(f"  - Development files should be in: {dev_folder}")
        sys.exit(1)
    
    print(f"\nStarting training with {len(train_files)} folds...")
    print(f"Configuration: {json.dumps(config, indent=2)}\n")
    
    # Track overall statistics
    start_time = time.time()
    results = {}
    
    # Process each fold
    for fold_number, (train_file, dev_file) in enumerate(zip(train_files, dev_files), start=1):
        fold_start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"Processing Fold {fold_number}/{len(train_files)}:")
        print(f"  Train File: {train_file}")
        print(f"  Dev File: {dev_file}")
        
        train_path = os.path.join(train_folder, train_file)
        dev_path = os.path.join(dev_folder, dev_file)
        output_dir = os.path.join(output_base_dir, f"fold{fold_number}")
        
        print(f"  Output Directory: {output_dir}")
        
        # Train the fold
        success, monitor = train_fold(
            fold_number=fold_number,
            train_path=train_path,
            dev_path=dev_path,
            output_dir=output_dir,
            config=config
        )
        
        # Process the output in real-time
        while True:
            # Read output line by line
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
                
            if line:
                line = line.strip()
                print(f"    {line}")  # Print the line to console with indentation
                monitor.process_line(line)
                
                # Show progress indicators for better user feedback
                if "Counting training words" in line:
                    print(f"    📊 Preparing training data...")
                elif "Training pipeline" in line:
                    print(f"    🚀 Starting training...")
                elif "Early stopping on patience triggered" in line:
                    print(f"    ⏹️ Training completed (early stopping)")
                elif "Saved model to output directory" in line:
                    print(f"    💾 Model saved successfully")
                
                # Break if there was an error
                if monitor.error:
                    print(f"❌ Error detected: {monitor.error}")
                    break
    # Save overall results
    total_time = time.time() - start_time
    results["summary"] = {
        "total_time": total_time,
        "total_folds": len(train_files),
        "successful_folds": sum(1 for r in results.values() if isinstance(r, dict) and r.get("success", False))
    }
    
    results_file = os.path.join(output_base_dir, "training_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Training completed in {total_time:.2f} seconds.")
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()