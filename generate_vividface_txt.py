import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import sys
import os

def generate_keypoints(video_path, txt_path):
    app = FaceAnalysis(name='buffalo_l', root='insightface_model', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    lines = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        faces = app.get(frame)
        if len(faces) == 0:
            lines.append("0,0,0,0,0,0,0,0,0,0,0,0,0,0")
            continue
            
        # Get the largest face
        face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
        
        # Bbox: x1, y1, x2, y2
        bbox = [int(v) for v in face.bbox]
        
        # Landmarks: 5 points * 2
        kps = face.kps
        kps_flat = [int(v) for pt in kps for v in pt]
        
        line = ",".join(map(str, bbox + kps_flat))
        lines.append(line)
        
    cap.release()
    
    with open(txt_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"Saved {len(lines)} lines to {txt_path}")

if __name__ == "__main__":
    video_path = sys.argv[1]
    txt_path = sys.argv[2]
    generate_keypoints(video_path, txt_path)
