import os, sys, subprocess, uuid, csv, time, json, glob, shutil, cv2
from datetime import datetime

VENV_PY = os.path.abspath("venv/bin/python")

with open('simple_jobs.json', 'r') as f: jobs = json.load(f)
audio_pool = sorted(glob.glob("videos/02_passed/audio_pool/**/*.wav", recursive=True))

for job in jobs:
    if job['idx'] % 10 < 8: continue
    v_in = os.path.abspath(f"videos/02_passed/chunks_ready/{job['bin']}/{job['id']}.mp4")
    v_out = os.path.abspath(f"test_st_out.mp4")
    
    print(f"Testing SadTalker for {job['id']}")
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
    
    if face_found:
        t_a = os.path.abspath(f"latentsync_temp/{uuid.uuid4()}_st.wav")
        subprocess.run(["ffmpeg", "-y", "-i", audio_pool[job['idx'] % len(audio_pool)], "-ss", "0", "-t", "2.0", "-ar", "16000", t_a], capture_output=True)
        res = os.path.abspath(f"latentsync_temp/st_test")
        if os.path.exists(res): shutil.rmtree(res)
        os.makedirs(res, exist_ok=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.abspath("SadTalker")
        
        print("Running SadTalker command...")
        # RUNNING WITHOUT CAPTURING OUTPUT TO SEE REAL-TIME
        cmd = [VENV_PY, "SadTalker/inference.py", "--driven_audio", t_a, "--source_image", ref, "--result_dir", res, "--still", "--preprocess", "full", "--enhancer", "gfpgan"]
        print(f"Command: {' '.join(cmd)}")
        subprocess.run(cmd, env=env)
        
        m = glob.glob(f"{res}/**/*.mp4", recursive=True)
        if m: print(f"SUCCESS: {m[0]}")
        else: print("FAILED: No output mp4")
    else:
        print("FAILED: No face found")
    break
