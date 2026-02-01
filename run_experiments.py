import os
import yaml
import subprocess
import itertools

def main():
    # Load base config
    with open("./config.yaml", "r") as f:
        conf = yaml.safe_load(f)
    
    # Define experiment space
    datasets = ["NetEase", "iFashion", "Youshu"]
    # datasets = ["NetEase"] # For quick testing
    cagcn_types = ["jc", "sc", "lhn", "co"]
    # cagcn_types = ["jc"] # For quick testing
    trend_coeffs = [1.0] # Can expand later
    
    # Path to python interpreter
    # Use the current python interpreter executing this script
    import sys
    python_exe = sys.executable
    
    # Log directory
    os.makedirs("experiment_logs", exist_ok=True)
    
    for dataset, c_type, t_coeff in itertools.product(datasets, cagcn_types, trend_coeffs):
        print(f"=========================================================")
        print(f"Running Experiment: Dataset={dataset}, Type={c_type}, Coeff={t_coeff}")
        print(f"=========================================================")
        
        # We need to override config settings.
        # Since train.py loads config.yaml, we can either:
        # 1. Modify config.yaml dynamically (risky if parallel)
        # 2. Pass args to train.py (train.py only accepts limited args)
        # 3. Create temporary config files
        
        # The best way without modifying train.py heavily is to modify config.yaml temporarily
        # But wait, we already added code to utility.py to read 'cagcn_type' from conf.
        # And train.py reads conf from config.yaml.
        # Let's modify config.yaml in place for each run (sequential execution).
        
        # Read current config
        with open("./config.yaml", "r") as f:
            current_conf = yaml.safe_load(f)
            
        # Update specific dataset config
        if dataset in current_conf:
            current_conf[dataset]['cagcn_type'] = c_type
            current_conf[dataset]['trend_coeff'] = t_coeff
        else:
            print(f"Warning: Dataset {dataset} not found in config.yaml")
            continue
            
        # Write back to config.yaml
        with open("./config.yaml", "w") as f:
            yaml.dump(current_conf, f)
            
        # Run train.py
        # We allow stdout to flow to the console so user can see tqdm progress bar
        # train.py already saves critical metrics to log/ directory
        
        cmd = [
            python_exe, "train.py",
            "-g", "0",
            "-m", "MultiCBR",
            "-d", dataset,
            "-i", f"AutoExp_{c_type}"
        ]
        
        print(f"Starting process: {' '.join(cmd)}")
        
        # subprocess.run will wait for completion and output directly to console
        try:
            subprocess.run(cmd, check=True)
            print(f"SUCCESS: Experiment {dataset}_{c_type} finished.\n")
        except subprocess.CalledProcessError as e:
            print(f"FAILURE: Experiment {dataset}_{c_type} failed with exit code {e.returncode}.\n")
            # Continue to next experiment even if one fails
            continue


if __name__ == "__main__":
    main()
