import os, sys, subprocess, uuid, csv, time, json, glob, shutil
from datetime import datetime

MT_PY = os.path.abspath("MuseTalk/musetalk_venv/bin/python")
MT_CWD = os.path.abspath("MuseTalk")
MANIFEST = "videos/07_generation_logs/generation_manifest.csv"

def log(msg):
    print(f"[{datetime.now()}] {msg}")
    sys.stdout.flush()

log("Worker MuseTalk starting...")

if not os.path.exists('simple_jobs.json'):
    log("ERROR: simple_jobs.json missing")
    sys.exit(1)

with open('simple_jobs.json', 'r') as f: jobs = json.load(f)
audio_pool = sorted(glob.glob("videos/02_passed/audio_pool/**/*.wav", recursive=True))

log(f"Found {len(jobs)} jobs and {len(audio_pool)} audios")

for job in jobs:
    if job['idx'] % 10 < 4 or job['idx'] % 10 >= 8: continue
    
    v_out = os.path.abspath(f"videos/06_generation/bin_A_lipsync/musetalk/output/{job['id']}_fake_musetalk.mp4")
    if os.path.exists(v_out): continue
    
    v_in = os.path.abspath(f"videos/02_passed/chunks_ready/{job['bin']}/{job['id']}.mp4")
    if not os.path.exists(v_in): continue
    
    log(f"Starting job {job['id']}")
    start_t = time.time()
    
    # Prep audio
    t_a = os.path.abspath(f"MuseTalk/test_output/{uuid.uuid4()}_mt.wav")
    os.makedirs(os.path.dirname(t_a), exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-i", audio_pool[job['idx'] % len(audio_pool)], "-ss", "0", "-t", "2.0", "-ar", "16000", "-ac", "1", t_a], capture_output=True)
    
    # Prep config
    y = os.path.abspath(f"MuseTalk/task_{uuid.uuid4()}.yaml")
    with open(y, 'w') as f: f.write(f"task:\n video_path: \"{v_in}\"\n audio_path: \"{t_a}\"\n")
    
    # Prep result dir
    r = os.path.abspath(f"MuseTalk/res_{uuid.uuid4()}")
    os.makedirs(r, exist_ok=True)
    
    # Run Inference
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}:{MT_CWD}"
    
    log(f"Running MuseTalk inference for {job['id']}...")
    res = subprocess.run([MT_PY, "-u", "scripts/inference.py", "--inference_config", y, "--use_float16", "--result_dir", r], cwd=MT_CWD, env=env, capture_output=True, text=True)
    
    if res.returncode != 0:
        log(f"Inference FAILED for {job['id']}. Return code: {res.returncode}")
        if res.stderr:
            log(f"Stderr: {res.stderr[-500:]}")
    
    # Move output
    found = glob.glob(f"{r}/**/*.mp4", recursive=True)
    status = 'FAILED'
    if found:
        shutil.move(found[0], v_out)
        log(f"SUCCESS: {v_out}")
        status = 'COMPLETED'
    else:
        log(f"No output found for {job['id']}")
        
    # Cleanup
    if os.path.exists(r): shutil.rmtree(r)
    if os.path.exists(y): os.remove(y)
    if os.path.exists(t_a): os.remove(t_a)
    
    with open(MANIFEST, 'a', newline='') as f:
        csv.writer(f).writerow([uuid.uuid4(), job['id'], 'musetalk', status, time.time()-start_t, datetime.now().isoformat()])
