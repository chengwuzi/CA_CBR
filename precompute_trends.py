import os
import itertools
import yaml
import torch
from utility import Datasets

def get_user_choice():
    print("\n" + "="*50)
    print("MultiCBR Trend Precomputer - Interactive Mode")
    print("="*50)
    
    # Select Dataset
    datasets = ["NetEase", "iFashion", "Youshu"]
    print("\nAvailable Datasets:")
    for i, d in enumerate(datasets):
        print(f"  [{i+1}] {d}")
    print(f"  [{len(datasets)+1}] ALL Datasets")
    
    while True:
        try:
            d_idx = int(input("\nSelect Dataset (Input number): ")) - 1
            if d_idx == len(datasets):
                selected_datasets = datasets
                break
            if 0 <= d_idx < len(datasets):
                selected_datasets = [datasets[d_idx]]
                break
            print("Invalid selection. Try again.")
        except ValueError:
            print("Please input a number.")

    # Select Metric
    cagcn_types = ["jc", "sc", "lhn", "co"]
    print("\nAvailable Metrics:")
    for i, m in enumerate(cagcn_types):
        print(f"  [{i+1}] {m.upper()}")
    print(f"  [{len(cagcn_types)+1}] ALL Metrics")
    
    while True:
        try:
            m_idx = int(input("\nSelect Metric (Input number): ")) - 1
            if m_idx == len(cagcn_types):
                selected_metrics = cagcn_types
                break
            if 0 <= m_idx < len(cagcn_types):
                selected_metrics = [cagcn_types[m_idx]]
                break
            print("Invalid selection. Try again.")
        except ValueError:
            print("Please input a number.")
            
    return selected_datasets, selected_metrics

def precompute_trends():
    """
    Standalone script to precompute CIR trends (jc, sc, lhn, co) for all datasets.
    Run this on a machine with good CPU/RAM (or GPU) to generate .pt files.
    Then upload the 'datasets' folder to your training server.
    """
    
    # Get user choices
    target_datasets, target_metrics = get_user_choice()
    
    print("\n" + "-"*50)
    print(f"Plan: Process {target_datasets} with metrics {target_metrics}")
    print("-"*50 + "\n")
    
    # Load base config to initialize Datasets class
    with open("./config.yaml", "r") as f:
        base_conf = yaml.safe_load(f)

    for dataset_name in target_datasets:
        print(f"\n{'='*50}")
        print(f"Processing Dataset: {dataset_name}")
        print(f"{'='*50}")
        
        # Construct a minimal conf for Datasets init
        conf = base_conf[dataset_name]
        conf['dataset'] = dataset_name
        # Force device to CUDA if available for the Dataset class init
        # Although utility.get_cir_trend does its own check, Datasets init might use it?
        # No, Datasets init just loads graphs.
        # But let's set it to cuda:0 to be consistent.
        conf['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
        
        # Initialize Dataset (loads graphs)
        dataset = Datasets(conf)
        
        # Calculate trends for all types
        for c_type in target_metrics:
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

    print("\n\nAll selected trends computed! Please copy the 'datasets' folder to your server.")

if __name__ == "__main__":
    precompute_trends()
