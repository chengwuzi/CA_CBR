import sys
import os
import platform

def check_environment():
    print("="*50)
    print("Environment Diagnostic Tool")
    print("="*50)
    
    print(f"Python Version: {sys.version}")
    print(f"OS: {platform.system()} {platform.release()}")
    print("-" * 30)

    # 1. Check Numpy (Crucial for version < 2.0)
    try:
        import numpy
        print(f"[OK] numpy: {numpy.__version__}")
        if int(numpy.__version__.split('.')[0]) >= 2:
            print("  [WARNING] numpy version is >= 2.0. This might cause compatibility issues with PyTorch/Scipy.")
    except ImportError as e:
        print(f"[FAIL] numpy: Not installed or error ({e})")

    # 2. Check PyTorch and CUDA
    try:
        import torch
        print(f"[OK] torch: {torch.__version__}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  GPU Count: {torch.cuda.device_count()}")
            print(f"  Current Device: {torch.cuda.get_device_name(0)}")
        else:
            print("  [WARNING] CUDA is NOT available. Training will be very slow on CPU.")
    except ImportError as e:
        print(f"[FAIL] torch: Not installed or error ({e})")

    # 3. Check Scipy
    try:
        import scipy
        print(f"[OK] scipy: {scipy.__version__}")
    except ImportError as e:
        print(f"[FAIL] scipy: Not installed or error ({e})")

    # 4. Check Optional torch-scatter
    try:
        import torch_scatter
        print(f"[OK] torch-scatter: {torch_scatter.__version__}")
    except ImportError:
        print(f"[INFO] torch-scatter: Not installed (Code will use slower fallback)")
    except Exception as e:
        print(f"[FAIL] torch-scatter: Error during import ({e})")

    print("-" * 30)
    
    # 5. Check Project Imports
    print("Checking Project Imports...")
    try:
        from utility import Datasets
        print("[OK] utility.py loaded successfully")
    except Exception as e:
        print(f"[FAIL] utility.py load failed: {e}")

    try:
        from models.MultiCBR import MultiCBR
        print("[OK] models/MultiCBR.py loaded successfully")
    except Exception as e:
        print(f"[FAIL] models/MultiCBR.py load failed: {e}")
        
    print("="*50)
    print("Diagnostic Finished.")

if __name__ == "__main__":
    check_environment()
