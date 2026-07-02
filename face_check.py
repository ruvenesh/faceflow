import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize MediaPipe Face Detector (v2 Tasks API)
# Note: This requires a model file (blaze_face_short_range.tflite)
# Since we don't have it yet, we will download it if it's missing.

def has_face(video_path: str, sample_frames: int = 8) -> bool:
    """
    Sample N evenly-spaced frames from the video.
    Return True if any frame contains a detected face using MediaPipe Tasks.
    """
    import os
    import requests

    model_path = 'face_landmarker.task'
    if not os.path.exists(model_path):
        print(f"Downloading MediaPipe model to {model_path}...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        r = requests.get(url, allow_redirects=True)
        open(model_path, 'wb').write(r.content)

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return False

    step = max(1, total_frames // sample_frames)
    found = False

    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = detector.detect(mp_image)
        
        if detection_result.face_landmarks:
            found = True
            break

    cap.release()
    return found