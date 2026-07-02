import os

BASE_DIR = os.path.abspath(".")

WORKERS = {
    "worker_phase1_inswapper.py": {
        "out": "worker_inswapper_bulk.py",
        "manifest": "videos/07_generation_logs/inswapper_bulk_manifest.csv",
        "log": "videos/07_generation_logs/inswapper_bulk.log",
        "replacements": {
            "df['model'] == 'inswapper'": "df['model_name'] == 'inswapper'",
            "job['ident']": "job['ident']"
        }
    },
    "worker_phase3_simswap.py": {
        "out": "worker_simswap_bulk.py",
        "manifest": "videos/07_generation_logs/simswap_bulk_manifest.csv",
        "log": "videos/07_generation_logs/simswap_bulk.log",
        "replacements": {}
    },
    "worker_phase4_facedancer.py": {
        "out": "worker_facedancer_bulk.py",
        "manifest": "videos/07_generation_logs/facedancer_bulk_manifest.csv",
        "log": "videos/07_generation_logs/facedancer_bulk.log",
        "replacements": {}
    },
    "worker_phase4_vividface.py": {
        "out": "worker_vividface_bulk.py",
        "manifest": "videos/07_generation_logs/vividface_bulk_manifest.csv",
        "log": "videos/07_generation_logs/vividface_bulk.log",
        "replacements": {}
    },
    "worker_phase2_dreamid.py": {
        "out": "worker_dreamid_bulk.py",
        "manifest": "videos/07_generation_logs/dreamid_bulk_manifest.csv",
        "log": "videos/07_generation_logs/dreamid_bulk.log",
        "replacements": {
            "df['model'] == 'dreamid-v'": "df['model_name'] == 'dreamid'"
        }
    }
}

def run():
    for src, config in WORKERS.items():
        with open(src, "r", encoding='utf-8') as f:
            content = f.read()
            
        # Update manifest and log
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith("MANIFEST = "):
                lines[i] = f'MANIFEST = os.path.join(BASE_DIR, "{config["manifest"]}")'
            elif line.startswith("LOG_FILE = "):
                lines[i] = f'LOG_FILE = os.path.join(BASE_DIR, "{config["log"]}")'
                
        content = '\n'.join(lines)
        
        # Apply custom replacements
        for old, new in config["replacements"].items():
            content = content.replace(old, new)
            
        # Write new worker
        with open(config["out"], "w", encoding='utf-8') as f:
            f.write(content)
            
        print(f"Generated {config['out']}")

if __name__ == "__main__":
    run()
