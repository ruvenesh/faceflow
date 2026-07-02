import os
import json
import sqlite3
import subprocess
import logging
import time
from datetime import datetime
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import requests
import random

import yaml

# ─── CONFIG ───────────────────────────────────────────────────────────────────
def load_config():
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    return {}

_cfg = load_config()

LIMIT_GB        = _cfg.get("global", {}).get("limit_gb", 10)
LIMIT_BYTES     = LIMIT_GB * 1024 ** 3
MIN_DURATION_S  = _cfg.get("global", {}).get("min_duration_s", 15)
MAX_DURATION_S  = _cfg.get("global", {}).get("max_duration_s", 600)
TARGET_DURATION_S = _cfg.get("global", {}).get("target_duration_s", 30)
VIDEOS_DIR      = _cfg.get("paths", {}).get("videos_dir", "videos")
DB_PATH         = _cfg.get("paths", {}).get("tiktok_db", "db/tiktok_pipeline.db")
LOG_PATH        = "logs/tiktok_pipeline.log"
COOKIES_FILE    = _cfg.get("paths", {}).get("cookies_file", "cookies.txt")
MODEL_PATH      = _cfg.get("global", {}).get("model_path", "face_landmarker.task")

# ─── AUTOMATIC QUERY ROTATION ──────────────────────────────────────────────────
TIKTOK_HASHTAGS = _cfg.get("tiktok", {}).get("hashtags", [
    "malaysia", "malaysiatiktok", "sembang", "podcastmalaysia"
])


def generate_random_queries(count=5):
    queries = []
    for _ in range(count):
        tag = random.choice(TIKTOK_HASHTAGS)
        query = f"https://www.tiktok.com/tag/{tag}"
        queries.append(query)
    return queries

# ─── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [TIKTOK] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ─── FACE DETECTION SETUP ─────────────────────────────────────────────────────

def get_detector():
    if not os.path.exists(MODEL_PATH):
        log.info(f"Downloading MediaPipe model to {MODEL_PATH}...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        r = requests.get(url, allow_redirects=True)
        with open(MODEL_PATH, 'wb') as f:
            f.write(r.content)

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(options)

# ─── DB & STATS ────────────────────────────────────────────────────────────────

def init_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            video_id      TEXT PRIMARY KEY,
            url           TEXT,
            status        TEXT,
            file_path     TEXT,
            file_size_mb  REAL,
            title         TEXT,
            channel       TEXT,
            duration_s    REAL,
            processed_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

def already_processed(conn, video_id: str) -> bool:
    row = conn.execute("SELECT status FROM videos WHERE video_id = ?", (video_id,)).fetchone()
    return row is not None

def mark(conn, video_id, status, url="", file_path="", file_size_mb=0, title="", channel="", duration_s=0):
    conn.execute("""
        INSERT OR REPLACE INTO videos
            (video_id, url, status, file_path, file_size_mb,
             title, channel, duration_s, processed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (video_id, url, status, file_path, file_size_mb, title, channel, duration_s, datetime.now()))
    conn.commit()

def get_used_bytes() -> int:
    total = 0
    if not os.path.exists(VIDEOS_DIR): return 0
    for fname in os.listdir(VIDEOS_DIR):
        fpath = os.path.join(VIDEOS_DIR, fname)
        if os.path.isfile(fpath): total += os.path.getsize(fpath)
    return total

# ─── DOWNLOAD ─────────────────────────────────────────────────────────────────

def download_video(url: str, video_id: str) -> str | None:
    # Prefix tiktok_ to ensure no collision with YT videos
    temp_id = f"tiktok_{video_id}"
    output_template = os.path.join(VIDEOS_DIR, f"{temp_id}_temp.%(ext)s")
    cmd = [
        "yt-dlp",
        "--format", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]",
        "--merge-output-format", "mp4",
        "--max-filesize", "250M",
        "--no-playlist",
        "--no-overwrites",
        "--output", output_template,
        "--no-warnings",
    ]
    if os.path.exists(COOKIES_FILE): cmd += ["--cookies", COOKIES_FILE]
    cmd.append(url)

    subprocess.run(cmd, capture_output=True)
    expected_path = os.path.join(VIDEOS_DIR, f"{temp_id}_temp.mp4")
    if os.path.exists(expected_path): return expected_path
    
    # Fallback to check if it downloaded with a slightly different name
    for fname in os.listdir(VIDEOS_DIR):
        if fname.startswith(f"{temp_id}_temp"):
            return os.path.join(VIDEOS_DIR, fname)
            
    return None

# ─── CROP & FACE DETECTION ────────────────────────────────────────────────────

def find_best_face_window(detector, video_path: str, target_s: int = 30) -> float | None:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0: return None
    duration = total_frames / fps
    
    # If the video is shorter than our target (e.g., a 20s TikTok), 
    # we just want to ensure there is a face for most of it.
    actual_target = min(duration, target_s)
    if duration < 5: # Too short to be useful
        cap.release()
        return None

    face_map = []
    scan_duration = min(int(duration), 300)
    for s in range(0, scan_duration):
        cap.set(cv2.CAP_PROP_POS_MSEC, s * 1000)
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = detector.detect(mp_image)
        face_map.append(1 if results.face_landmarks else 0)
    
    cap.release()

    # Search for the best window
    window_size = int(actual_target)
    if window_size <= 0: return None
    
    for i in range(len(face_map) - window_size + 1):
        window = face_map[i : i + window_size]
        if sum(window) / window_size > 0.8:
            return float(i)
            
    return None

def crop_video(input_path: str, output_path: str, start_s: float, duration_s: int = 30):
    cmd = [
        "ffmpeg", "-y", "-ss", str(start_s), "-i", input_path,
        "-t", str(duration_s), "-c:v", "libx264", "-c:a", "aac",
        "-strict", "experimental", output_path
    ]
    subprocess.run(cmd, capture_output=True)

# ─── MAIN RUNNER ───────────────────────────────────────────────────────────────

def run():
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    os.makedirs("db", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    detector = get_detector()
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    used_bytes = get_used_bytes()
    log.info(f"PIPELINE START. Target: {LIMIT_GB}GB. Currently at: {used_bytes/1024**3:.2f}GB")

    while used_bytes < LIMIT_BYTES:
        queries = generate_random_queries(count=5)
        log.info(f"Generated new batch of {len(queries)} TikTok hashtag queries.")

        for query in queries:
            if used_bytes >= LIMIT_BYTES: break
            
            log.info(f"── Scraping Hashtag: {query}")
            
            # Use yt-dlp to extract up to 30 video URLs from the hashtag page
            cmd = ["yt-dlp", "--flat-playlist", "--print", "%(id)s\t%(duration)s\t%(title)s\t%(webpage_url)s", "--playlist-end", "30", query]
            if os.path.exists(COOKIES_FILE): cmd += ["--cookies", COOKIES_FILE]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            entries = result.stdout.strip().splitlines()
            log.info(f"Found {len(entries)} videos in {query}")
            
            for line in entries:
                if used_bytes >= LIMIT_BYTES: break

                parts = line.split("\t")
                if len(parts) < 4: continue
                video_id, dur_raw, title, url = parts
                
                try: dur = float(dur_raw)
                except: dur = 0
                
                if already_processed(conn, video_id): continue
                if dur > 0 and (dur < MIN_DURATION_S or dur > MAX_DURATION_S): continue

                log.info(f"Processing: {video_id} | {title[:50]}")
                temp_path = download_video(url, video_id)
                if not temp_path:
                    mark(conn, video_id, "error")
                    continue

                start_time = find_best_face_window(detector, temp_path, TARGET_DURATION_S)
                
                if start_time is None:
                    os.remove(temp_path)
                    mark(conn, video_id, "rejected_no_face")
                    continue

                final_path = os.path.join(VIDEOS_DIR, f"tiktok_{video_id}.mp4")
                crop_video(temp_path, final_path, start_time, TARGET_DURATION_S)
                if os.path.exists(temp_path): os.remove(temp_path)
                
                if os.path.exists(final_path):
                    fsize = os.path.getsize(final_path)
                    used_bytes += fsize
                    mark(conn, video_id, "kept", url=url, file_path=final_path, 
                         file_size_mb=fsize/1024**2, title=title, duration_s=30)
                    log.info(f"KEPT ✅ tiktok_{video_id} | Total: {used_bytes/1024**3:.2f}GB / {LIMIT_GB}GB")
                else:
                    mark(conn, video_id, "error_crop")

        log.info("Batch complete. Sleeping to avoid TikTok rate limits...")
        time.sleep(45)

    log.info("🎯 TARGET REACHED: 10GB attained. Pipeline stopping.")
    conn.close()

if __name__ == "__main__":
    run()