import cv2
import numpy as np
from .face import extract_normed_embedding

def extract_best_frame_from_video(video_bytes: bytes):
    # Save to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    best_emb = None
    best_frame = None
    best_bbox = None
    max_confidence = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        emb, bbox = extract_normed_embedding(frame)
        if emb is not None:
            # Use similarity norm as "confidence"
            conf = np.linalg.norm(emb)
            if conf > max_confidence:
                max_confidence = conf
                best_emb = emb
                best_frame = frame
                best_bbox = bbox

    cap.release()
    return best_frame, best_emb, best_bbox
