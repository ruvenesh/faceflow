import os, sys, subprocess, uuid, csv, time, json, glob, shutil, cv2
from datetime import datetime

VENV_PY = os.path.abspath("venv/bin/python")
SAD_DIR = os.path.abspath("SadTalker")
MANIFEST = "videos/07_generation_logs/generation_manifest.csv"

def log(msg):
    print(f"[{datetime.now()}] {msg}")
    sys.stdout.flush()

log("Worker SadTalker starting...")

if not os.path.exists('simple_jobs.json'):
    log("ERROR: simple_jobs.json missing")
    sys.exit(1)

with open('simple_jobs.json', 'r') as f: jobs = json.load(f)
audio_pool = sorted(glob.glob("videos/02_passed/audio_pool/**/*.wav", recursive=True))

for job in jobs:
    if job['idx'] % 10 < 8: continue # 20% Target
    
    v_out = os.path.abspath(f"videos/06_generation/bin_A_lipsync/sadtalker/output/{job['id']}_fake_sadtalker.mp4")
    if os.path.exists(v_out): continue
    
    v_in = os.path.abspath(f"videos/02_passed/chunks_ready/{job['bin']}/{job['id']}.mp4")
    if not os.path.exists(v_in): continue
    
    log(f"Starting job {job['id']}")
    start_t = time.time()
    
    ref = os.path.abspath(f"latentsync_temp/{uuid.uuid4()}_ref.jpg")
    os.makedirs(os.path.dirname(ref), exist_ok=True)
    face_found = False
    cap = cv2.VideoCapture(v_in)
    for frame_idx in [30, 0, 15, 45, 59]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(ref, frame)
            face_found = True
            break
    cap.release()
    
    status = 'FAILED'
    if face_found:
        t_a = os.path.abspath(f"latentsync_temp/{uuid.uuid4()}_st.wav")
        subprocess.run(["ffmpeg", "-y", "-i", audio_pool[job['idx'] % len(audio_pool)], "-ss", "0", "-t", "2.0", "-ar", "16000", "-ac", "1", t_a], capture_output=True)
        
        res = os.path.abspath(f"latentsync_temp/st_{uuid.uuid4()}")
        os.makedirs(res, exist_ok=True)
        
        env = os.environ.copy()
        env["PYTHONPATH"] = SAD_DIR
        
        log(f"Running SadTalker inference for {job['id']}...")
        # RUN FROM SADTALKER DIR TO FIX RELATIVE PATHS
        sub_res = subprocess.run([VENV_PY, "inference.py", "--driven_audio", t_a, "--source_image", ref, "--result_dir", res, "--still", "--preprocess", "full", "--enhancer", "gfpgan"], cwd=SAD_DIR, capture_output=True, env=env, text=True)
        
        m = glob.glob(f"{res}/**/*.mp4", recursive=True)
        if m:
            shutil.move(m[0], v_out)
            log(f"SUCCESS: {v_out}")
            status = 'COMPLETED'
        else:
            log(f"No output found for {job['id']}")
            if sub_res.stderr:
                log(f"Stderr: {sub_res.stderr[-500:]}")
            
        if os.path.exists(res): shutil.rmtree(res)
        if os.path.exists(t_a): os.remove(t_a)
        if os.path.exists(ref): os.remove(ref)
    else:
        log(f"No usable frames found for {job['id']}")

    with open(MANIFEST, 'a', newline='') as f:
        csv.writer(f).writerow([uuid.uuid4(), job['id'], 'sadtalker', status, time.time()-start_t, datetime.now().isoformat()])
