import os, sys, subprocess, time
from datetime import datetime

BASE_DIR = os.path.abspath(".")
LOG_FILE = os.path.join(BASE_DIR, "videos/07_generation_logs/master_binCD_bulk.log")

WORKERS = [
    # "worker_inswapper_bulk.py",
    "worker_simswap_bulk.py",
    "worker_facedancer_bulk.py",
    "worker_vividface_bulk.py",
    "worker_dreamid_bulk.py"
]

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"[{datetime.now()}] {msg}\n")
    print(f"[{datetime.now()}] {msg}")

def run():
    log("=== Master Bin CD Bulk Orchestrator Started ===")
    
    for worker_script in WORKERS:
        worker_path = os.path.join(BASE_DIR, worker_script)
        if not os.path.exists(worker_path):
            log(f"ERROR: Worker script not found: {worker_path}")
            continue
            
        log(f"\n--- Starting {worker_script} ---")
        start_t = time.time()
        
        try:
            # Run native python. Individual workers will invoke WSL if they need it.
            cmd = [sys.executable, worker_script]
                
            # Capture stdout and stderr
            res = subprocess.run(cmd, cwd=BASE_DIR, text=True)
            
            if res.returncode != 0:
                log(f"WARNING: {worker_script} exited with code {res.returncode}")
            else:
                log(f"SUCCESS: {worker_script} finished smoothly.")
                
        except Exception as e:
            log(f"FATAL ERROR running {worker_script}: {e}")
            
        duration = (time.time() - start_t) / 3600
        log(f"--- Finished {worker_script} (Duration: {duration:.2f} hours) ---\n")
        
        # Cool down and release memory
        time.sleep(10)
        
    log("=== Master Bin CD Bulk Orchestrator Finished ===")

if __name__ == "__main__":
    run()
