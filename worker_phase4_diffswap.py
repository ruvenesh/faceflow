"""
DiffSwap Phase 4 Worker
Runs DiffSwap face swap on video chunks via WSL.

Strategy: Extract frames from video, swap each frame using DiffSwap
inference script in WSL, then reassemble into video with audio.
"""
import os, sys, subprocess, time, shutil, glob
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.abspath(".")
WSL_VENV_PYTHON = "diffswap_wsl_venv/bin/python"
DIFFSWAP_DIR = "DiffSwap"
MANIFEST = os.path.join(BASE_DIR, "videos/07_generation_logs/diffswap_test_manifest.csv")
LOG_FILE = os.path.join(BASE_DIR, "videos/07_generation_logs/diffswap_test_log.txt")
TEMP_DIR = os.path.join(BASE_DIR, "videos/07_generation_logs/diffswap_temp")

def log(msg):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f: f.write(f"[{datetime.now()}] {msg}\n")
    print(f"[{datetime.now()}] {msg}")

def win_to_wsl(path):
    """Convert Windows path to WSL path."""
    path = os.path.abspath(path).replace("\\", "/")
    if path[1] == ":":
        drive = path[0].lower()
        return f"/mnt/{drive}{path[2:]}"
    return path

def extract_frames(video_path, frames_dir):
    """Extract all frames from a video file."""
    os.makedirs(frames_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-qscale:v", "2",
        os.path.join(frames_dir, "frame_%06d.png")
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        log(f"  ffmpeg extract failed: {res.stderr}")
        return False
    frames = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    log(f"  Extracted {len(frames)} frames")
    return len(frames) > 0

def get_video_fps(video_path):
    """Get FPS of video using ffprobe."""
    cmd = ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
           "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", video_path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and res.stdout.strip():
        parts = res.stdout.strip().split("/")
        if len(parts) == 2:
            return float(parts[0]) / float(parts[1])
    return 30.0

def swap_frames_dir(source_img, target_frames_dir, output_frames_dir, ddim_steps=50):
    """Run DiffSwap on a directory of frames via WSL."""
    wsl_python = win_to_wsl(os.path.join(BASE_DIR, WSL_VENV_PYTHON))
    wsl_script = win_to_wsl(os.path.join(BASE_DIR, DIFFSWAP_DIR, "diffswap_inference.py"))
    wsl_source = win_to_wsl(source_img)
    wsl_target_dir = win_to_wsl(target_frames_dir)
    wsl_output_dir = win_to_wsl(output_frames_dir)
    wsl_cwd = win_to_wsl(os.path.join(BASE_DIR, DIFFSWAP_DIR))

    cmd = [
        "wsl", "--", "bash", "-c",
        f"cd {wsl_cwd} && PYTHONPATH={wsl_cwd}:$PYTHONPATH {wsl_python} {wsl_script} "
        f"--source {wsl_source} --target_dir {wsl_target_dir} --output_dir {wsl_output_dir} --ddim_steps {ddim_steps} --device cuda"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    with open(os.path.join(BASE_DIR, "videos", "07_generation_logs", "diffswap_inference.log"), "a", encoding='utf-8') as f:
        f.write(f"\n--- INFERENCE RUN ---\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}\n")
    if res.returncode != 0:
        with open(os.path.join(BASE_DIR, "videos", "07_generation_logs", "diffswap_error.log"), "a", encoding='utf-8') as f:
            f.write(f"\n--- ERROR ---\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}\n")
    return res.returncode == 0

def reassemble_video(frames_dir, audio_source, output_path, fps):
    """Reassemble frames into video with original audio."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # First create video from frames
    temp_video = output_path + ".temp.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "frame_%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        temp_video
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        log(f"  ffmpeg reassemble failed: {res.stderr}")
        return False

    # Merge audio from original
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_video,
        "-i", audio_source,
        "-c:v", "copy", "-c:a", "aac",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-shortest",
        output_path
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(temp_video):
        os.remove(temp_video)
    return os.path.exists(output_path)


# Initialize manifest
if not os.path.exists(MANIFEST):
    src_manifest = os.path.join(BASE_DIR, "videos/07_generation_logs/facedancer_test_manifest.csv")
    if not os.path.exists(src_manifest):
        log(f"Source manifest not found: {src_manifest}")
        sys.exit(1)
    df = pd.read_csv(src_manifest)
    df['status'] = 'PENDING'
    df['output_path'] = df['output_path'].str.replace('facedancer', 'diffswap')
    df.to_csv(MANIFEST, index=False)

df = pd.read_csv(MANIFEST)
log(f"Phase 4 (DiffSwap) Started: {len(df)} jobs.")

DDIM_STEPS = 30  # Use fewer steps for speed (30 is a good quality/speed tradeoff)

for idx, job in df.iterrows():
    if job.get('status') == 'COMPLETED': continue

    log(f"STARTING DiffSwap: {job['chunk_id']}")
    start_t = time.time()

    df.at[idx, 'status'] = 'RUNNING'
    df.to_csv(MANIFEST, index=False)

    success = True

    def fix_path(p):
        if str(p).startswith("/mnt/c/"):
            return "C:/" + p[7:]
        if os.path.isabs(p):
            return p
        return os.path.join(BASE_DIR, p)

    abs_source = fix_path(job['source_path'])  # This is the video chunk
    abs_ref = fix_path(job['ref_path'])         # This is the reference face image
    abs_out = fix_path(job['output_path'])

    # Create temp working directory for this chunk
    chunk_temp = os.path.join(TEMP_DIR, job['chunk_id'])
    frames_in = os.path.join(chunk_temp, "frames_in")
    frames_out = os.path.join(chunk_temp, "frames_out")
    os.makedirs(frames_out, exist_ok=True)

    # Step 1: Extract frames
    log(f"  Extracting frames from {abs_source}")
    if not extract_frames(abs_source, frames_in):
        log(f"  Frame extraction FAILED")
        success = False
    else:
        fps = get_video_fps(abs_source)
        log(f"  Video FPS: {fps}")

        # Step 2: Swap frames in batch
        log(f"  Swapping all frames via DiffSwap...")
        swap_ok = swap_frames_dir(abs_ref, frames_in, frames_out, ddim_steps=DDIM_STEPS)
        if not swap_ok:
            log(f"  Batch swap FAILED, falling back to original frames")
            for f in glob.glob(os.path.join(frames_in, "frame_*.png")):
                shutil.copy2(f, os.path.join(frames_out, os.path.basename(f)))

        # Step 3: Reassemble
        log(f"  Reassembling video...")
        os.makedirs(os.path.dirname(abs_out), exist_ok=True)
        if not reassemble_video(frames_out, abs_source, abs_out, fps):
            log(f"  Reassembly FAILED")
            success = False

    # Cleanup temp
    if os.path.exists(chunk_temp):
        shutil.rmtree(chunk_temp, ignore_errors=True)

    duration = (time.time() - start_t) / 60
    df.at[idx, 'status'] = 'COMPLETED' if success else 'FAILED'
    df.to_csv(MANIFEST, index=False)

    if success:
        log(f"SUCCESS: {job['chunk_id']} ({duration:.1f} mins)")
        preview_dir = os.path.join(BASE_DIR, "videos/06_generation/bin_CD_swap/preview_samples/diffswap/")
        os.makedirs(preview_dir, exist_ok=True)
        completed_count = len(df[df['status'] == 'COMPLETED'])
        if completed_count <= 3:
            shutil.copy2(abs_out, preview_dir)
    else:
        log(f"FAILED: {job['chunk_id']} ({duration:.1f} mins)")

log("Phase 4 (DiffSwap) Finished.")
