import os
import sys
import glob

def check_prereqs():
    print("--- Running Prerequisite Checks ---")
    
    # Check 1
    dir1 = "videos/02_passed/chunks_ready/bin_B_reenactment"
    mp4s1 = glob.glob(os.path.join(dir1, "*.mp4"))
    if len(mp4s1) < 20:
        print(f"FAIL: {dir1} contains {len(mp4s1)} mp4s, expected >= 20.")
        sys.exit(1)
    print(f"PASS: {dir1} contains {len(mp4s1)} mp4s.")
    
    # Check 2
    dir2 = "videos/06_generation/bin_B_reenactment/driver_pool"
    mp4s2 = glob.glob(os.path.join(dir2, "*.mp4"))
    if len(mp4s2) < 20:
        print(f"FAIL: {dir2} contains {len(mp4s2)} mp4s, expected >= 20.")
        sys.exit(1)
    print(f"PASS: {dir2} contains {len(mp4s2)} mp4s.")
    
    # Check 3
    manifest = "videos/05_logs/chunk_manifest.csv"
    if not os.path.exists(manifest):
        print(f"FAIL: {manifest} does not exist.")
        sys.exit(1)
    print(f"PASS: {manifest} exists.")
    
    # Check 4
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        if not torch.cuda.is_available():
            print("FAIL: CUDA is not available.")
            sys.exit(1)
        name = torch.cuda.get_device_name(0)
        print(f"CUDA Device: {name}")
        if "5080" not in name:
            print(f"FAIL: Expected RTX 5080, got {name}")
            sys.exit(1)
        print("PASS: CUDA is available and RTX 5080 is detected.")
    except ImportError:
        print("FAIL: PyTorch is not installed in the current environment.")
        sys.exit(1)
        
    print("--- All Prerequisite Checks Passed ---")

if __name__ == "__main__":
    check_prereqs()
