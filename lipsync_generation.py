import os
import sys
import subprocess
import shutil
import uuid
import csv
import json
import time
import threading
from pathlib import Path
from datetime import datetime

vram_lock = threading.Lock()
VENV_PY = os.path.abspath("venv/bin/python")

def append_to_csv(filepath, row):
    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f); writer.writerow([str(x) for x in row])

def trim_audio(audio_in, duration, audio_out):
    subprocess.run(["ffmpeg", "-y", "-i", str(audio_in), "-ss", "0", "-t", str(duration), "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(audio_out)], capture_output=True)

def run_wav2lip(job, logs_dir):
    v_in, a_in, v_out = job['v_in'], job['a_in'], job['v_out']
    start_t = time.time()
    try:
        subprocess.run([VENV_PY, "Wav2Lip/inference.py", "--checkpoint_path", "Wav2Lip/checkpoints/wav2lip_gan.pth", "--face", v_in, "--audio", str(a_in), "--outfile", v_out, "--pads", "0", "10", "0", "0", "--nosmooth"], capture_output=True, timeout=300)
        status = 'COMPLETED' if os.path.exists(v_out) else 'FAILED'
    except Exception as e: status = f'ERROR: {str(e)}'
    append_to_csv(logs_dir / "generation_manifest.csv", [uuid.uuid4(), job['chunk_id'], 'wav2lip', status, time.time()-start_t, datetime.now().isoformat()])

def run_musetalk(job, logs_dir):
    with vram_lock:
        v_in, a_in, v_out = job['v_in'], job['a_in'], job['v_out']
        t_audio = Path(f"MuseTalk/test_output/{uuid.uuid4()}_mt.wav").absolute()
        task_yaml = Path(f"MuseTalk/task_{uuid.uuid4()}.yaml").absolute()
        start_t = time.time()
        try:
            trim_audio(a_in, 2.0, t_audio)
            with open(task_yaml, 'w') as f: f.write(f"task:\n video_path: \"{v_in}\"\n audio_path: \"{t_audio}\"\n")
            mt_py = os.path.abspath("MuseTalk/musetalk_venv/bin/python")
            mt_cwd = os.path.abspath("MuseTalk")
            env = os.environ.copy(); env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}:{mt_cwd}"
            res_dir = Path(f"MuseTalk/res_{uuid.uuid4()}").absolute(); res_dir.mkdir(exist_ok=True, parents=True)
            subprocess.run([mt_py, "-u", "scripts/inference.py", "--inference_config", str(task_yaml), "--use_float16", "--result_dir", str(res_dir)], cwd=mt_cwd, env=env, capture_output=True, timeout=300)
            found = list(res_dir.rglob("*.mp4"))
            if found: shutil.move(str(found[0]), v_out); shutil.rmtree(res_dir)
            status = 'COMPLETED' if os.path.exists(v_out) else 'FAILED'
        except Exception as e: status = f'ERROR: {str(e)}'
        finally:
            if task_yaml.exists(): os.remove(task_yaml)
            if t_audio.exists(): os.remove(t_audio)
        append_to_csv(logs_dir / "generation_manifest.csv", [uuid.uuid4(), job['chunk_id'], 'musetalk', status, time.time()-start_t, datetime.now().isoformat()])
        time.sleep(5)

def run_sadtalker(job, logs_dir):
    with vram_lock:
        v_in, a_in, v_out = job['v_in'], job['a_in'], job['v_out']
        ref = Path(f"latentsync_temp/{uuid.uuid4()}_ref.jpg").absolute()
        t_audio = Path(f"latentsync_temp/{uuid.uuid4()}_st.wav").absolute()
        start_t = time.time()
        try:
            cap = cv2.VideoCapture(v_in); cap.set(cv2.CAP_PROP_POS_FRAMES, 30); ret, frame = cap.read(); cap.release()
            if ret: cv2.imwrite(str(ref), frame)
            else: raise Exception("No frame 30")
            trim_audio(a_in, 2.0, t_audio)
            env = os.environ.copy(); env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}:{os.path.abspath('SadTalker')}"
            res_dir = Path(f"latentsync_temp/st_{uuid.uuid4()}").absolute()
            subprocess.run([VENV_PY, "SadTalker/inference.py", "--driven_audio", str(t_audio), "--source_image", str(ref), "--result_dir", str(res_dir), "--still", "--preprocess", "full", "--enhancer", "gfpgan"], capture_output=True, env=env, timeout=600)
            m = list(res_dir.rglob("*.mp4"))
            if m: shutil.move(str(m[0]), v_out); shutil.rmtree(res_dir)
            status = 'COMPLETED' if os.path.exists(v_out) else 'FAILED'
        except Exception as e: status = f'ERROR: {str(e)}'
        finally:
            if ref.exists(): os.remove(ref)
            if t_audio.exists(): os.remove(t_audio)
        append_to_csv(logs_dir / "generation_manifest.csv", [uuid.uuid4(), job['chunk_id'], 'sadtalker', status, time.time()-start_t, datetime.now().isoformat()])
        time.sleep(5)

def thread_worker(jobs, model_type, logs_dir):
    for job in jobs:
        if job['model_name'] != model_type: continue
        if os.path.exists(job['v_out']): continue
        try:
            if model_type == 'wav2lip': run_wav2lip(job, logs_dir)
            elif model_type == 'musetalk': run_musetalk(job, logs_dir)
            elif model_type == 'sadtalker': run_sadtalker(job, logs_dir)
        except: pass

def main():
    try:
        with open('job_queue.json', 'r') as f: jobs = json.load(f)
        logs_dir = Path("videos/07_generation_logs")
        threads = []
        for mtype in ['wav2lip', 'musetalk', 'sadtalker']:
            t = threading.Thread(target=thread_worker, args=(jobs, mtype, logs_dir), daemon=True)
            t.start(); threads.append(t)
        while any(t.is_alive() for t in threads): time.sleep(10)
    except Exception as e:
        with open('FATAL_CRASH.txt', 'w') as f: f.write(str(e))

if __name__ == "__main__":
    import cv2
    main()
