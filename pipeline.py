import os
os.environ["GLOG_minloglevel"] = "2"
import os
import sys
import argparse
import subprocess
import shutil
import uuid
from datetime import datetime
import csv
from pathlib import Path
import multiprocessing
import math

# 1.2 Install Required Dependencies
def install_dependencies():
    packages = [
        "opencv-python<4.13", "scenedetect[opencv]", "ultralytics", "mediapipe<0.10.20",
        "ffmpeg-python", "webrtcvad", "librosa", "imagehash", "Pillow",
        "pandas", "tqdm", "numpy<2", "scipy", "omegaconf"
    ]
    print("Installing/verifying dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])

install_dependencies()

import cv2
from scenedetect import detect, ContentDetector
from ultralytics import YOLO
import mediapipe as mp
import ffmpeg
import webrtcvad
import librosa
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

def check_system():
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg is not available as a system binary. Please install it and halt.")
        sys.exit(1)
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. GPU phases will fall back to CPU and performance will be severely degraded.")

def build_directory_tree(videos_dir: Path):
    dirs = [
        "00_raw_intake",
        "01_processing_workspace/chunks",
        "01_processing_workspace/audio",
        "02_passed/chunks_ready/bin_A_lipsync",
        "02_passed/chunks_ready/bin_B_reenactment",
        "02_passed/chunks_ready/bin_C_gan_swap",
        "02_passed/chunks_ready/bin_D_diffusion_swap",
        "02_passed/chunks_ready/bin_E_reference_only",
        "02_passed/chunks_ready/bin_A_lipsync_rescued",
        "02_passed/chunks_ready/bin_B_reenactment_rescued",
        "02_passed/chunks_ready/bin_C_gan_swap_rescued",
        "02_passed/chunks_ready/bin_D_diffusion_swap_rescued",
        "02_passed/chunks_ready/bin_E_reference_only_rescued",
        "02_passed/audio_pool/by_identity",
        "03_rejected/phase1_integrity",
        "03_rejected/phase2_scene_cuts",
        "03_rejected/phase3_face_spatial/no_face",
        "03_rejected/phase3_face_spatial/multi_face",
        "03_rejected/phase3_face_spatial/face_too_small",
        "03_rejected/phase4_pose_trait",
        "03_rejected/phase5_audio",
        "04_rescue_pipeline/tracklet_workspace",
        "04_rescue_pipeline/superres_queue",
        "04_rescue_pipeline/rescue_crops",
        "04_rescue_pipeline/rescue_rejected",
        "05_logs"
    ]
    for d in dirs:
        (videos_dir / d).mkdir(parents=True, exist_ok=True)

def append_to_csv(filepath, row):
    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def init_manifests(logs_dir: Path):
    master = logs_dir / "master_manifest.csv"
    if not master.exists():
        append_to_csv(master, [
            "video_id", "original_filename", "source_path", "intake_timestamp",
            "total_frames", "measured_fps", "duration_seconds", "resolution",
            "codec", "final_status", "rejection_phase", "rejection_reason",
            "chunk_ids", "bin_assignment", "identity_group_id", "audio_status",
            "rescue_eligible", "rescue_status", "rescue_clip_ids", "split_assignment", "notes"
        ])
    chunk = logs_dir / "chunk_manifest.csv"
    if not chunk.exists():
        append_to_csv(chunk, [
            "chunk_id", "parent_video_id", "chunk_index", "frame_start", "frame_end",
            "scene_cut_found", "phase2_status", "phase3_status", "phase3_rejection_reason",
            "face_detection_min_conf", "face_bbox_min_size", "multi_face_detected",
            "phase4_status", "phase4_rejection_reason", "lip_variance_score",
            "head_variance_score", "face_quality_score", "assigned_bin",
            "routing_confidence", "final_chunk_status"
        ])
    rej = logs_dir / "rejection_log.csv"
    if not rej.exists():
        append_to_csv(rej, ["video_id", "chunk_id", "phase", "reason", "timestamp"])
    bin_log = logs_dir / "bin_assignment_log.csv"
    if not bin_log.exists():
        append_to_csv(bin_log, ["chunk_id", "parent_video_id", "lip_variance_score", "head_variance_score", "face_quality_score", "assigned_bin", "routing_confidence", "timestamp"])
    rescue = logs_dir / "rescue_manifest.csv"
    if not rescue.exists():
        append_to_csv(rescue, ["rescue_clip_id", "parent_video_id", "track_id", "frames_tracked", "bbox_min_size", "continuity_score", "phase3_status", "final_bin_assigned", "audio_usable", "rejection_reason"])

def update_master_manifest(logs_dir, video_id, updates):
    df = pd.read_csv(logs_dir / "master_manifest.csv", dtype=str)
    idx = df.index[df['video_id'] == video_id]
    if not idx.empty:
        for k, v in updates.items():
            df.loc[idx, k] = str(v)
        df.to_csv(logs_dir / "master_manifest.csv", index=False)

def update_chunk_manifest(logs_dir, chunk_id, updates):
    df = pd.read_csv(logs_dir / "chunk_manifest.csv", dtype=str)
    idx = df.index[df['chunk_id'] == chunk_id]
    if not idx.empty:
        for k, v in updates.items():
            df.loc[idx, k] = str(v)
        df.to_csv(logs_dir / "chunk_manifest.csv", index=False)

def get_master_df(logs_dir):
    return pd.read_csv(logs_dir / "master_manifest.csv", dtype=str)

def get_chunk_df(logs_dir):
    return pd.read_csv(logs_dir / "chunk_manifest.csv", dtype=str)

def do_intake(videos_dir: Path, limit=None):
    logs_dir = videos_dir / "05_logs"
    intake_dir = videos_dir / "00_raw_intake"
    map_path = logs_dir / "filename_map.csv"
    
    if not map_path.exists():
        append_to_csv(map_path, ["original_filename", "uuid_filename"])
    
    existing_map = pd.read_csv(map_path)
    processed_originals = set(existing_map["original_filename"].tolist())

    supported_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    videos_found = [p for p in videos_dir.iterdir() if p.is_file() and p.suffix.lower() in supported_exts]
    
    if limit is not None:
        videos_found = videos_found[:limit]

    print(f"Intake: Found {len(videos_found)} videos")
    df_master = get_master_df(logs_dir)
    
    for vid in tqdm(videos_found, desc="Copying to intake"):
        if vid.name in processed_originals:
            continue
            
        uid = str(uuid.uuid4())
        new_name = f"{uid}{vid.suffix.lower()}"
        dest = intake_dir / new_name
        shutil.copy2(vid, dest)
        append_to_csv(map_path, [vid.name, new_name])
        
        cap = cv2.VideoCapture(str(dest))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        measured_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        cap.release()
        
        duration = total_frames / measured_fps if measured_fps > 0 else 0
        notes = "fps_mismatch" if measured_fps != 30.0 else ""

        append_to_csv(logs_dir / "master_manifest.csv", [
            uid, vid.name, str(vid.absolute()), datetime.utcnow().isoformat(),
            total_frames, measured_fps, duration, f"{width}x{height}",
            codec, "PENDING", "", "", "", "", "", "", "", "", "", "", notes
        ])

def phase1(videos_dir: Path):
    logs_dir = videos_dir / "05_logs"
    intake_dir = videos_dir / "00_raw_intake"
    workspace_dir = videos_dir / "01_processing_workspace"
    rej_dir = videos_dir / "03_rejected/phase1_integrity"
    rej_log = logs_dir / "rejection_log.csv"
    
    df = get_master_df(logs_dir)
    pending = df[df['final_status'] == 'PENDING']
    
    print(f"Phase 1: Processing {len(pending)} pending videos")
    for _, row in tqdm(pending.iterrows(), total=len(pending)):
        uid = row['video_id']
        ext = Path(row['original_filename']).suffix.lower()
        src = intake_dir / f"{uid}{ext}"
        
        def fail(reason):
            update_master_manifest(logs_dir, uid, {'final_status': 'REJECTED', 'rejection_phase': 1, 'rejection_reason': reason})
            shutil.copy2(src, rej_dir / f"{uid}{ext}")
            append_to_csv(rej_log, [uid, "", 1, reason, datetime.utcnow().isoformat()])

        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            fail("corrupt_header")
            continue
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 180:
            fail("insufficient_frames")
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret1, f1 = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(30, total_frames-1))
        ret30, f30 = cap.read()
        cap.release()
        
        if ret1 and ret30:
            g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
            g30 = cv2.cvtColor(f30, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(g1, g30)
            if np.mean(diff) < 2.0:
                fail("static_frames")
                continue
        
        shutil.copy2(src, workspace_dir / f"{uid}{ext}")
        update_master_manifest(logs_dir, uid, {'final_status': 'PHASE1_PASSED'})

def phase2(videos_dir: Path):
    logs_dir = videos_dir / "05_logs"
    workspace_dir = videos_dir / "01_processing_workspace"
    chunk_dir = workspace_dir / "chunks"
    audio_dir = workspace_dir / "audio"
    rej_dir = videos_dir / "03_rejected/phase2_scene_cuts"
    rej_log = logs_dir / "rejection_log.csv"
    
    df = get_master_df(logs_dir)
    passed1 = df[df['final_status'] == 'PHASE1_PASSED']
    
    print(f"Phase 2: Processing {len(passed1)} videos")
    for _, row in tqdm(passed1.iterrows(), total=len(passed1)):
        uid = row['video_id']
        ext = Path(row['original_filename']).suffix.lower()
        src = workspace_dir / f"{uid}{ext}"
        
        if not src.exists(): continue
        
        audio_out = audio_dir / f"{uid}.wav"
        if not audio_out.exists():
            try:
                (ffmpeg.input(str(src)).output(str(audio_out), acodec='pcm_s16le', ac=1, ar='16k').overwrite_output().run(quiet=True))
            except ffmpeg.Error:
                update_master_manifest(logs_dir, uid, {'audio_status': 'no_audio_stream'})
        
        cap = cv2.VideoCapture(str(src))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        chunk_idx = 1
        chunks_passed = 0
        
        for start_f in range(0, total_frames - 59, 60):
            end_f = start_f + 59
            chunk_id = str(uuid.uuid4())
            chunk_path = chunk_dir / f"{chunk_id}.mp4"
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            writer = cv2.VideoWriter(str(chunk_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            for _ in range(60):
                ret, frame = cap.read()
                if not ret: break
                writer.write(frame)
            writer.release()
            
            cut_list = detect(str(chunk_path), ContentDetector(threshold=30.0))
            cut_found = len(cut_list) > 0
            
            status = 'REJECTED' if cut_found else 'PASSED'
            if not cut_found: chunks_passed += 1
            else: append_to_csv(rej_log, [uid, chunk_id, 2, "scene_cut", datetime.utcnow().isoformat()])
            
            append_to_csv(logs_dir / "chunk_manifest.csv", [
                chunk_id, uid, chunk_idx, start_f, end_f, cut_found, status, 
                "PENDING" if not cut_found else "", "", "", "", "", "", "", "", "", "", "", "", "PENDING"
            ])
            chunk_idx += 1
            
        cap.release()
        
        if chunks_passed == 0:
            update_master_manifest(logs_dir, uid, {'final_status': 'REJECTED', 'rejection_phase': 2, 'rejection_reason': 'all_chunks_have_scene_cuts'})
            shutil.copy2(src, rej_dir / f"{uid}{ext}")
        else:
            update_master_manifest(logs_dir, uid, {'final_status': 'PHASE2_PASSED'})

def phase3(videos_dir: Path):
    logs_dir = videos_dir / "05_logs"
    chunk_dir = videos_dir / "01_processing_workspace/chunks"
    rej_dir_base = videos_dir / "03_rejected/phase3_face_spatial"
    rej_log = logs_dir / "rejection_log.csv"

    df_chunk = get_chunk_df(logs_dir)
    pending_chunks = df_chunk[df_chunk['phase2_status'] == 'PASSED']
    pending_chunks = pending_chunks[pending_chunks['phase3_status'] == 'PENDING']
    
    if len(pending_chunks) == 0:
        return
        
    print(f"Phase 3: Processing {len(pending_chunks)} chunks")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO("yolov8n-face.pt")
    
    for _, row in tqdm(pending_chunks.iterrows(), total=len(pending_chunks)):
        chunk_id = row['chunk_id']
        parent_id = row['parent_video_id']
        chunk_path = chunk_dir / f"{chunk_id}.mp4"
        
        if not chunk_path.exists(): continue
        
        cap = cv2.VideoCapture(str(chunk_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()
        
        if len(frames) == 0:
            continue
            
        # Batched inference
        results = model.predict(frames, device=device, verbose=False)
        
        reject_reason = None
        needs_superres = False
        multi_face = False
        min_conf = 1.0
        sum_conf = 0.0
        frame_count = 0
        min_bbox_size = float('inf')
        
        frame_h, frame_w = frames[0].shape[:2]
        absolute_min_px = int(frame_h * 0.15)
        
        for res in results:
            boxes = res.boxes
            if len(boxes) == 0:
                reject_reason = "no_face_detected"
                break
            elif len(boxes) > 1:
                confidences = boxes.conf.cpu().numpy()
                valid_faces = sum(c >= 0.50 for c in confidences)
                if valid_faces > 1:
                    reject_reason = "multi_face_detected"
                    multi_face = True
                    break
                    
            if len(boxes) > 0:
                best_idx = torch.argmax(boxes.conf).item()
                best_box = boxes[best_idx]
                conf = best_box.conf.item()
                
                min_conf = min(min_conf, conf)
                sum_conf += conf
                frame_count += 1
                
                # Use bounding box height as primary metric
                h = best_box.xywh[0][3].item()
                w = best_box.xywh[0][2].item()
                min_bbox_size = min(min_bbox_size, w, h)
                
                if h < absolute_min_px:
                    reject_reason = "face_too_small"
                    break
                    
        avg_conf = (sum_conf / frame_count) if frame_count > 0 else 0.0

        if reject_reason is None and avg_conf < 0.60:
            reject_reason = "low_average_confidence"

        if reject_reason:
            update_chunk_manifest(logs_dir, chunk_id, {
                'phase3_status': 'REJECTED',
                'phase3_rejection_reason': reject_reason,
                'face_detection_min_conf': min_conf,
                'face_bbox_min_size': min_bbox_size,
                'multi_face_detected': multi_face
            })
            append_to_csv(rej_log, [parent_id, chunk_id, 3, reject_reason, datetime.utcnow().isoformat()])
            
            if reject_reason == "multi_face_detected":
                shutil.copy2(chunk_path, rej_dir_base / "multi_face" / f"{chunk_id}.mp4")
                update_master_manifest(logs_dir, parent_id, {'rescue_eligible': True, 'rescue_status': 'PENDING'})
            elif reject_reason == "face_too_small":
                shutil.copy2(chunk_path, rej_dir_base / "face_too_small" / f"{chunk_id}.mp4")
            elif reject_reason == "no_face_detected":
                shutil.copy2(chunk_path, rej_dir_base / "no_face" / f"{chunk_id}.mp4")
            else:
                shutil.copy2(chunk_path, rej_dir_base / f"{chunk_id}.mp4") # Fallback
        else:
            update_chunk_manifest(logs_dir, chunk_id, {
                'phase3_status': 'PASSED',
                'face_detection_min_conf': min_conf,
                'face_bbox_min_size': min_bbox_size,
                'multi_face_detected': multi_face,
                'phase4_status': 'PENDING'
            })

def process_phase4_chunk(args):
    import os
    os.environ["GLOG_minloglevel"] = "3"  # Suppress MediaPipe C++ warnings
    import mediapipe as mp
    chunk_path_str, chunk_id = args
    
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    
    cap = cv2.VideoCapture(chunk_path_str)
    
    yaws = []
    pitches = []
    lip_dists = []
    max_yaw = 0
    max_pitch = 0
    mouth_occluded = False
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h_dim, w_dim, _ = frame.shape
        
        # Hand check for occlusion
        hand_results = hands.process(rgb)
        hand_landmarks = []
        if hand_results.multi_hand_landmarks:
            for hand_landmarks_i in hand_results.multi_hand_landmarks:
                for lm in hand_landmarks_i.landmark:
                    hand_landmarks.append((int(lm.x * w_dim), int(lm.y * h_dim)))
        
        face_results = face_mesh.process(rgb)
        if face_results.multi_face_landmarks:
            face_lms = face_results.multi_face_landmarks[0]
            
            # Extract Mouth Bbox
            mouth_indices = [13, 14, 78, 308, 0, 17] # basic mouth bounding
            m_x = [face_lms.landmark[i].x * w_dim for i in mouth_indices]
            m_y = [face_lms.landmark[i].y * h_dim for i in mouth_indices]
            mx_min, mx_max = min(m_x), max(m_x)
            my_min, my_max = min(m_y), max(m_y)
            
            # Check Occlusion
            for hx, hy in hand_landmarks:
                if mx_min <= hx <= mx_max and my_min <= hy <= my_max:
                    mouth_occluded = True
                    break
            
            # Pose Estimation (Simplified PnP)
            face_3d = []
            face_2d = []
            for idx, lm in enumerate(face_lms.landmark):
                if idx in [33, 263, 1, 61, 291, 199]:
                    x, y = int(lm.x * w_dim), int(lm.y * h_dim)
                    face_2d.append([x, y])
                    face_3d.append([lm.x, lm.y, lm.z])
            
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)
            focal_length = 1 * w_dim
            cam_matrix = np.array([[focal_length, 0, h_dim / 2], [0, focal_length, w_dim / 2], [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, jac = cv2.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            
            pitch = abs(angles[0])
            yaw = abs(angles[1])
            
            yaws.append(yaw)
            pitches.append(pitch)
            max_yaw = max(max_yaw, yaw)
            max_pitch = max(max_pitch, pitch)
            
            # Lip distance
            upper_lip = face_lms.landmark[13]
            lower_lip = face_lms.landmark[14]
            dist = math.sqrt((upper_lip.x - lower_lip.x)**2 + (upper_lip.y - lower_lip.y)**2) * h_dim
            lip_dists.append(dist)
            
    cap.release()
    face_mesh.close()
    hands.close()
    
    return chunk_id, max_yaw, max_pitch, mouth_occluded, np.std(lip_dists) if lip_dists else 0, np.std(yaws) if yaws else 0

def phase4(videos_dir: Path):
    logs_dir = videos_dir / "05_logs"
    chunk_dir = videos_dir / "01_processing_workspace/chunks"
    rej_dir = videos_dir / "03_rejected/phase4_pose_trait"
    rej_log = logs_dir / "rejection_log.csv"
    bin_log = logs_dir / "bin_assignment_log.csv"

    df_chunk = get_chunk_df(logs_dir)
    pending_chunks = df_chunk[df_chunk['phase4_status'] == 'PENDING']
    
    if len(pending_chunks) == 0:
        return
        
    print(f"Phase 4: Processing {len(pending_chunks)} chunks")
    
    rescue_manifest_path = logs_dir / "rescue_manifest.csv"
    rescued_chunks = set()
    if rescue_manifest_path.exists():
        df_rescue = pd.read_csv(rescue_manifest_path, dtype=str)
        rescued_chunks = set(df_rescue[df_rescue['phase3_status'] == 'PASSED_RESCUED']['rescue_clip_id'].tolist())
    
    tasks = []
    for _, row in pending_chunks.iterrows():
        chunk_path = chunk_dir / f"{row['chunk_id']}.mp4"
        if chunk_path.exists():
            tasks.append((str(chunk_path), row['chunk_id']))

    with multiprocessing.Pool(processes=min(8, multiprocessing.cpu_count()), maxtasksperchild=10) as pool:
        results = list(tqdm(pool.imap(process_phase4_chunk, tasks), total=len(tasks)))

    for chunk_id, max_yaw, max_pitch, mouth_occluded, lip_var, head_var in results:
        row = df_chunk[df_chunk['chunk_id'] == chunk_id].iloc[0]
        parent_id = row['parent_video_id']
        face_qual = float(row['face_detection_min_conf'])
        
        reject_reason = None
        if mouth_occluded: reject_reason = "mouth_occluded"
        
        if reject_reason:
            update_chunk_manifest(logs_dir, chunk_id, {'phase4_status': 'REJECTED', 'phase4_rejection_reason': reject_reason})
            append_to_csv(rej_log, [parent_id, chunk_id, 4, reject_reason, datetime.utcnow().isoformat()])
            shutil.copy2(chunk_dir / f"{chunk_id}.mp4", rej_dir / f"{chunk_id}.mp4")
            continue
            
        assigned_bin = "bin_E_reference_only"
        routing_conf = "low"
        
        if lip_var >= 2.5 and head_var <= 4.16:
            assigned_bin = "bin_A_lipsync"
            routing_conf = "high"
        elif head_var <= 2.77 and lip_var < 2.5:
            assigned_bin = "bin_B_reenactment"
            routing_conf = "high"
        elif face_qual >= 0.72 and head_var > 2.77:
            assigned_bin = "bin_C_gan_swap"
            routing_conf = "medium"
        elif face_qual >= 0.68 and head_var > 4.16:
            assigned_bin = "bin_D_diffusion_swap"
            routing_conf = "medium"
            
        if chunk_id in rescued_chunks:
            assigned_bin += "_rescued"
            
        update_chunk_manifest(logs_dir, chunk_id, {
            'phase4_status': 'PASSED',
            'lip_variance_score': lip_var,
            'head_variance_score': head_var,
            'assigned_bin': assigned_bin,
            'routing_confidence': routing_conf,
            'final_chunk_status': 'PENDING'
        })
        append_to_csv(bin_log, [chunk_id, parent_id, lip_var, head_var, face_qual, assigned_bin, routing_conf, datetime.utcnow().isoformat()])

def phase5(videos_dir: Path):
    logs_dir = videos_dir / "05_logs"
    audio_dir = videos_dir / "01_processing_workspace/audio"
    audio_pool = videos_dir / "02_passed/audio_pool/by_identity"
    rej_dir = videos_dir / "03_rejected/phase5_audio"
    
    df_master = get_master_df(logs_dir)
    df_chunk = get_chunk_df(logs_dir)
    
    passed_chunks = df_chunk[df_chunk['phase4_status'] == 'PASSED']
    parent_ids = passed_chunks['parent_video_id'].unique()
    
    print(f"Phase 5: Processing audio for {len(parent_ids)} videos")
    
    vad = webrtcvad.Vad(2)
    
    for vid in tqdm(parent_ids):
        audio_path = audio_dir / f"{vid}.wav"
        if not audio_path.exists(): continue
        
        try:
            y, sr = librosa.load(str(audio_path), sr=16000)
            
            # RMS check
            rms = librosa.feature.rms(y=y)[0]
            mean_rms = np.mean(rms)
            mean_db = 20 * np.log10(mean_rms) if mean_rms > 0 else -100
            
            if mean_db < -40:
                reject_audio(vid, "low_audio_energy", audio_path, rej_dir, logs_dir)
                continue
                
            # Clipping check
            if np.sum(np.abs(y) >= 0.99) / len(y) > 0.01:
                reject_audio(vid, "audio_clipping", audio_path, rej_dir, logs_dir)
                continue
                
            # VAD check
            y_int16 = (y * 32767).astype(np.int16)
            frame_duration_ms = 30
            n_samples = int(sr * frame_duration_ms / 1000)
            
            silence_frames = 0
            total_vad_frames = 0
            for i in range(0, len(y_int16) - n_samples, n_samples):
                frame = y_int16[i:i+n_samples].tobytes()
                if not vad.is_speech(frame, sr):
                    silence_frames += 1
                total_vad_frames += 1
                
            if total_vad_frames > 0 and silence_frames / total_vad_frames > 0.5:
                reject_audio(vid, "excessive_silence", audio_path, rej_dir, logs_dir)
                continue
                
            # Passed
            ident_dir = audio_pool / vid
            ident_dir.mkdir(exist_ok=True)
            shutil.copy2(audio_path, ident_dir / f"{vid}.wav")
            update_master_manifest(logs_dir, vid, {'audio_status': 'verified'})
            
        except Exception as e:
            reject_audio(vid, "audio_processing_error", audio_path, rej_dir, logs_dir)

def reject_audio(vid, reason, audio_path, rej_dir, logs_dir):
    shutil.copy2(audio_path, rej_dir / f"{vid}.wav")
    update_master_manifest(logs_dir, vid, {'audio_status': f"rejected_{reason}"})
    
    df_chunk = get_chunk_df(logs_dir)
    chunks = df_chunk[(df_chunk['parent_video_id'] == vid) & (df_chunk['assigned_bin'] == 'bin_A_lipsync')]
    for _, row in chunks.iterrows():
        update_chunk_manifest(logs_dir, row['chunk_id'], {
            'assigned_bin': 'bin_E_reference_only',
            'routing_confidence': 'demoted_no_audio'
        })

def phase6(videos_dir: Path):
    logs_dir = videos_dir / "05_logs"
    chunk_dir = videos_dir / "01_processing_workspace/chunks"
    passed_dir = videos_dir / "02_passed/chunks_ready"
    
    df_master = get_master_df(logs_dir)
    df_chunk = get_chunk_df(logs_dir)
    
    passed_chunks = df_chunk[df_chunk['phase4_status'] == 'PASSED']
    
    print(f"Phase 6: Routing {len(passed_chunks)} final chunks")
    
    for _, row in tqdm(passed_chunks.iterrows(), total=len(passed_chunks)):
        chunk_id = row['chunk_id']
        bin_dir = row['assigned_bin']
        
        src = chunk_dir / f"{chunk_id}.mp4"
        dst = passed_dir / bin_dir / f"{chunk_id}.mp4"
        
        if src.exists():
            shutil.copy2(src, dst)
            update_chunk_manifest(logs_dir, chunk_id, {'final_chunk_status': 'DELIVERED'})

    # Refresh chunk dataframe to get updated DELIVERED statuses
    df_chunk = get_chunk_df(logs_dir)

    # Reconcile master manifest
    for _, row in df_master.iterrows():
        vid = row['video_id']
        vid_chunks = df_chunk[df_chunk['parent_video_id'] == vid]
        
        total_extracted = len(vid_chunks)
        passed = vid_chunks[vid_chunks['final_chunk_status'] == 'DELIVERED']
        passed_count = len(passed)
        rescue_el = str(row.get('rescue_eligible', '')) == 'True'
        
        if total_extracted > 0:
            chunk_ids_str = ",".join(passed['chunk_id'].tolist())
            update_master_manifest(logs_dir, vid, {'chunk_ids': chunk_ids_str})
            
            if passed_count == 0 and not rescue_el:
                update_master_manifest(logs_dir, vid, {'final_status': 'REJECTED'})
            elif passed_count == 0 and rescue_el:
                update_master_manifest(logs_dir, vid, {'final_status': 'PENDING_RESCUE'})
            elif passed_count > 0 and passed_count < total_extracted:
                update_master_manifest(logs_dir, vid, {'final_status': 'PARTIAL_PASS'})
            elif passed_count == total_extracted:
                update_master_manifest(logs_dir, vid, {'final_status': 'FULL_PASS'})

def generate_report(videos_dir: Path):
    logs_dir = videos_dir / "05_logs"
    df_master = get_master_df(logs_dir)
    df_chunk = get_chunk_df(logs_dir)
    
    total_ingested = len(df_master)
    full = len(df_master[df_master['final_status'] == 'FULL_PASS'])
    partial = len(df_master[df_master['final_status'] == 'PARTIAL_PASS'])
    rejected = len(df_master[df_master['final_status'] == 'REJECTED'])
    rescue = len(df_master[df_master['final_status'] == 'PENDING_RESCUE'])
    
    total_chunks = len(df_chunk)
    passed_chunks = len(df_chunk[df_chunk['final_chunk_status'] == 'DELIVERED'])
    
    bins = df_chunk[df_chunk['final_chunk_status'] == 'DELIVERED']['assigned_bin'].value_counts().to_dict()
    
    audio_pool = list((videos_dir / "02_passed/audio_pool/by_identity").rglob("*.wav"))
    
    multi_face = len(df_chunk[df_chunk['phase3_rejection_reason'] == 'multi_face_detected']['parent_video_id'].unique())
    too_small = len(df_chunk[df_chunk['phase3_rejection_reason'] == 'face_too_small']['parent_video_id'].unique())
    
    p1 = len(df_master[df_master['rejection_phase'] == 1])
    p2 = len(df_chunk[df_chunk['phase2_status'] == 'REJECTED'])
    p3_no = len(df_chunk[df_chunk['phase3_rejection_reason'] == 'no_face_detected'])
    p3_multi = len(df_chunk[df_chunk['phase3_rejection_reason'] == 'multi_face_detected'])
    p3_small = len(df_chunk[df_chunk['phase3_rejection_reason'] == 'face_too_small'])
    p3_lost = len(df_chunk[df_chunk['phase3_rejection_reason'] == 'face_lost_mid_chunk'])
    p3_avg_conf = len(df_chunk[df_chunk['phase3_rejection_reason'] == 'low_average_confidence'])
    p3_sr = len(df_chunk[df_chunk['phase3_status'] == 'PENDING_SR'])
    p4 = len(df_chunk[df_chunk['phase4_status'] == 'REJECTED'])
    p5 = len(df_chunk[df_chunk['routing_confidence'] == 'demoted_no_audio'])
    
    report = f"""=== PIPELINE SUMMARY ===
Run timestamp: {datetime.utcnow().isoformat()}
Total videos ingested: {total_ingested}
Total videos: {full} FULL_PASS / {partial} PARTIAL_PASS / {rejected} REJECTED / {rescue} PENDING_RESCUE
Total chunks extracted: {total_chunks}
Total chunks passed all phases: {passed_chunks}
Chunks by bin:
    bin_A_lipsync: {bins.get('bin_A_lipsync', 0)}
    bin_B_reenactment: {bins.get('bin_B_reenactment', 0)}
    bin_C_gan_swap: {bins.get('bin_C_gan_swap', 0)}
    bin_D_diffusion_swap: {bins.get('bin_D_diffusion_swap', 0)}
    bin_E_reference_only: {bins.get('bin_E_reference_only', 0)}
Audio pool size (verified .wav files): {len(audio_pool)}
Videos flagged for rescue (multi_face): {multi_face}
Videos flagged for rescue (face_too_small): {too_small}

Phase rejection breakdown:
    Phase 1 rejections: {p1}
    Phase 2 rejections (chunk level): {p2}
    Phase 3 - no face: {p3_no}
    Phase 3 - multi face: {p3_multi}
    Phase 3 - face lost mid-chunk (conf < 0.40): {p3_lost}
    Phase 3 - low avg confidence (< 0.80): {p3_avg_conf}
    Phase 3 - face too small (HARD REJECT): {p3_small}
    Phase 3 - sent to Super-Res Queue: {p3_sr}
    Phase 4 rejections: {p4}
    Phase 5 audio failures (demotions only): {p5}

Bin A demotions due to audio failure: {p5}
Estimated dataset size for ViT training (paired real+fake): {bins.get('bin_A_lipsync', 0) + bins.get('bin_B_reenactment', 0)}
"""
    
    with open(logs_dir / "pipeline_summary_report.txt", "w") as f:
        f.write(report)
        
    print("\n" + report)

def main():
    parser = argparse.ArgumentParser(description="Deepfake Detection Preprocessing Pipeline")
    parser.add_argument("videos_dir", help="Path to the videos directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of videos processed during intake")
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir).resolve()
    check_system()
    build_directory_tree(videos_dir)
    init_manifests(videos_dir / "05_logs")
    
    do_intake(videos_dir, limit=args.limit)
    phase1(videos_dir)
    phase2(videos_dir)
    phase3(videos_dir)
    phase4(videos_dir)
    phase5(videos_dir)
    phase6(videos_dir)
    generate_report(videos_dir)
    
    print("Pipeline run completed fully.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
