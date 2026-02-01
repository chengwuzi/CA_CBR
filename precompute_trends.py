import os
import itertools
import yaml
import torch
from utility import Datasets

def precompute_trends():
    """
    Standalone script to precompute CIR trends (jc, sc, lhn, co) for all datasets.
    Run this on a machine with good CPU/RAM (or GPU) to generate .pt files.
    Then upload the 'datasets' folder to your training server.
    """
    
    # Define experiment space
    datasets = ["NetEase", "iFashion", "Youshu"]
    cagcn_types = ["jc", "sc", "lhn", "co"]
    
    # Load base config to initialize Datasets class
    with open("./config.yaml", "r") as f:
        base_conf = yaml.safe_load(f)

    for dataset_name in datasets:
        print(f"\n{'='*50}")
        print(f"Processing Dataset: {dataset_name}")
        print(f"{'='*50}")
        
        # Construct a minimal conf for Datasets init
        conf = base_conf[dataset_name]
        conf['dataset'] = dataset_name
        conf['device'] = torch.device('cpu') # Force CPU or let utility decide
        # Note: utility.get_cir_trend will check cuda availability internally
        
        # Initialize Dataset (loads graphs)
        dataset = Datasets(conf)
        
        # Calculate trends for all types
        for c_type in cagcn_types:
            print(f"\n>> Calculating {c_type.upper()} trends for {dataset_name}...")
            
            # 1. UB Graph
            # Check if file exists to skip
            path_ub = os.path.join(dataset.path, dataset.name, f'trend_ub_{c_type}.pt')
            if os.path.exists(path_ub):
                print(f"   [Skipped] {path_ub} exists.")
            else:
                dataset.get_cir_trend(dataset.u_b_graph_train, c_type, 'ub')
                
            # 2. UI Graph
            path_ui = os.path.join(dataset.path, dataset.name, f'trend_ui_{c_type}.pt')
            if os.path.exists(path_ui):
                print(f"   [Skipped] {path_ui} exists.")
            else:
                dataset.get_cir_trend(dataset.u_i_graph_train, c_type, 'ui')
                
            # 3. BI Graph
            path_bi = os.path.join(dataset.path, dataset.name, f'trend_bi_{c_type}.pt')
            if os.path.exists(path_bi):
                print(f"   [Skipped] {path_bi} exists.")
            else:
                dataset.get_cir_trend(dataset.b_i_graph_train, c_type, 'bi')

    print("\n\nAll trends computed! Please copy the 'datasets' folder to your server.")

if __name__ == "__main__":
    precompute_trends()
