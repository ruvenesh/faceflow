import os, glob, random, csv

BASE_DIR = os.path.abspath(".")
CHUNKS_DIRS = [
    os.path.join(BASE_DIR, "videos/02_passed/chunks_ready/bin_C_gan_swap"),
    os.path.join(BASE_DIR, "videos/02_passed/chunks_ready/bin_C_gan_swap_rescued"),
    os.path.join(BASE_DIR, "videos/02_passed/chunks_ready/bin_D_diffusion_swap"),
    os.path.join(BASE_DIR, "videos/02_passed/chunks_ready/bin_D_diffusion_swap_rescued")
]
ID_POOL_DIR = os.path.join(BASE_DIR, "videos/08_identity_pool")
LOG_DIR = os.path.join(BASE_DIR, "videos/07_generation_logs")

TARGETS = {
    "inswapper": 3380,
    "simswap": 2028,
    "facedancer": 1352,
    "vividface": 2002,
    "dreamid": 1638
}

def run():
    all_chunks = []
    for d in CHUNKS_DIRS:
        all_chunks.extend(glob.glob(os.path.join(d, "*.mp4")))
    
    all_chunks = list(set(all_chunks))
    random.seed(42) # Deterministic shuffle
    random.shuffle(all_chunks)
    
    print(f"Total chunks found: {len(all_chunks)}")
    total_requested = sum(TARGETS.values())
    print(f"Total requested: {total_requested}")
    
    if len(all_chunks) < total_requested:
        print(f"WARNING: Not enough chunks! Missing {total_requested - len(all_chunks)}. Will truncate the last models.")
    
    identities = glob.glob(os.path.join(ID_POOL_DIR, "*.png"))
    if not identities:
        print("ERROR: No identities found!")
        return

    chunk_idx = 0
    
    for model, requested_count in TARGETS.items():
        manifest_path = os.path.join(LOG_DIR, f"{model}_bulk_manifest.csv")
        jobs = []
        
        count = 0
        while count < requested_count and chunk_idx < len(all_chunks):
            chunk_path = all_chunks[chunk_idx]
            chunk_id = os.path.basename(chunk_path).replace(".mp4", "")
            ident_path = random.choice(identities)
            ident_id = os.path.basename(ident_path).replace(".png", "")
            
            output_name = f"{chunk_id}_fake_swap_{model}.mp4"
            output_path = os.path.join(BASE_DIR, "videos", "06_generation", "bin_CD_swap", model, output_name)
            
            jobs.append([
                chunk_id,                 # job_id
                chunk_id,                 # chunk_id
                chunk_id,                 # parent_id
                model,                    # model (for inswapper)
                model,                    # model_name
                ident_id,                 # ident
                "PENDING",                # status
                chunk_path,               # source_path
                ident_path,               # ref_path
                output_path,              # output_path
                "50",                     # ddim_steps
                ""                        # fail_reason
            ])
            chunk_idx += 1
            count += 1
            
        with open(manifest_path, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["job_id", "chunk_id", "parent_id", "model", "model_name", "ident", "status", "source_path", "ref_path", "output_path", "ddim_steps", "fail_reason"])
            writer.writerows(jobs)
            
        print(f"Generated {manifest_path} with {count} jobs (requested {requested_count}).")

if __name__ == "__main__":
    run()
