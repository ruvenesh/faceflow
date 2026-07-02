import os, glob, random, csv

BASE_DIR = os.path.abspath(".")
CHUNKS_DIR = os.path.join(BASE_DIR, "videos/02_passed/chunks_ready/bin_D_diffusion_swap")
ID_POOL_DIR = os.path.join(BASE_DIR, "videos/08_identity_pool")
MANIFEST = os.path.join(BASE_DIR, "videos/07_generation_logs/bin_CD_phase5_manifest.csv")

def run():
    chunks = glob.glob(os.path.join(CHUNKS_DIR, "*.mp4"))
    identities = glob.glob(os.path.join(ID_POOL_DIR, "*.png"))
    
    if len(chunks) < 60:
        print(f"Warning: Only {len(chunks)} chunks available.")
    
    selected_chunks = chunks[:60]
    
    jobs = []
    models = ["ghost"] * 20 + ["vface"] * 20 + ["alphaface"] * 20
    
    for i, chunk_path in enumerate(selected_chunks):
        chunk_id = os.path.basename(chunk_path).replace(".mp4", "")
        ident_id = os.path.basename(random.choice(identities)).replace(".png", "")
        model = models[i] if i < len(models) else "ghost"
        
        jobs.append([chunk_id, chunk_id, model, ident_id, "PENDING"])
        
    with open(MANIFEST, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["chunk_id", "parent_id", "model", "ident", "status"])
        writer.writerows(jobs)
        
    print(f"Created phase 5 manifest with {len(jobs)} jobs at {MANIFEST}")

if __name__ == "__main__":
    run()
