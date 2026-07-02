import os, sys, subprocess, uuid, csv, time, json, glob
from datetime import datetime

VENV_PY = os.path.abspath("venv/bin/python")
MANIFEST = "videos/07_generation_logs/generation_manifest.csv"

def log(msg):
    print(f"[{datetime.now()}] {msg}")
    sys.stdout.flush()

log("Worker Wav2Lip starting...")

with open('simple_jobs.json', 'r') as f: jobs = json.load(f)
audio_pool = sorted(glob.glob("videos/02_passed/audio_pool/**/*.wav", recursive=True))

for job in jobs:
    if job['idx'] % 10 >= 4: continue # 40% Target
    
    v_out = os.path.abspath(f"videos/06_generation/bin_A_lipsync/wav2lip/output/{job['id']}_fake_wav2lip.mp4")
    if os.path.exists(v_out): continue
    
    v_in = os.path.abspath(f"videos/02_passed/chunks_ready/{job['bin']}/{job['id']}.mp4")
    if not os.path.exists(v_in): continue
    
    log(f"Starting job {job['id']}")
    start_t = time.time()
    
    t_a = f"latentsync_temp/{uuid.uuid4()}_w2l.wav"
    subprocess.run(["ffmpeg", "-y", "-i", audio_pool[job['idx'] % len(audio_pool)], "-ss", "0", "-t", "2.0", "-ar", "16000", "-ac", "1", t_a], capture_output=True)
    
    subprocess.run([VENV_PY, "Wav2Lip/inference.py", "--checkpoint_path", "Wav2Lip/checkpoints/wav2lip_gan.pth", "--face", v_in, "--audio", t_a, "--outfile", v_out, "--pads", "0", "10", "0", "0", "--nosmooth"], capture_output=True)
    
    if os.path.exists(t_a): os.remove(t_a)
    
    status = 'COMPLETED' if os.path.exists(v_out) else 'FAILED'
    with open(MANIFEST, 'a', newline='') as f:
        csv.writer(f).writerow([uuid.uuid4(), job['id'], 'wav2lip', status, time.time()-start_t, datetime.now().isoformat()])
