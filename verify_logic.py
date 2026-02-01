import torch
import numpy as np
import scipy.sparse as sp
import os
import shutil
from utility import Datasets

class MockDatasets(Datasets):
    def __init__(self):
        # Bypass original init
        self.path = "./temp_verify"
        self.name = "VerifyDataset"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Running verification on: {self.device}")
        
        os.makedirs(os.path.join(self.path, self.name), exist_ok=True)

def verify():
    print("="*50)
    print("VERIFYING CALCULATION LOGIC")
    print("="*50)
    
    # 1. Create a asymmetric random graph (User=100, Item=200)
    # This ensures we catch any dimension mismatch errors (like the 18528 vs 22864 one)
    n_users = 100
    n_items = 200
    
    # Create random edges
    row = np.random.randint(0, n_users, 1000)
    col = np.random.randint(0, n_items, 1000)
    data = np.ones(1000)
    
    graph = sp.csr_matrix((data, (row, col)), shape=(n_users, n_items))
    print(f"Created random graph: {graph.shape}")
    
    # 2. Instantiate Mock Dataset
    dataset = MockDatasets()
    
    # 3. Run get_cir_trend (The function you are worried about)
    try:
        # We test 'jc' metric as it involves degree normalization (most prone to dimension bugs)
        print("\n>> Testing 'jc' calculation (simulating UB graph)...")
        trend = dataset.get_cir_trend(graph, 'jc', 'test_ub')
        
        print("\n[SUCCESS] Calculation finished without error!")
        print(f"Result shape: {trend.shape}")
        print(f"Expected shape: ({n_users+n_items}, {n_users+n_items})")
        
        if trend.shape == (n_users+n_items, n_users+n_items):
            print(">> Shape Check: PASS")
        else:
            print(">> Shape Check: FAIL")
            
    except Exception as e:
        print(f"\n[FAIL] Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if os.path.exists("./temp_verify"):
            shutil.rmtree("./temp_verify")
            print("\nCleaned up temp files.")

if __name__ == "__main__":
    verify()
