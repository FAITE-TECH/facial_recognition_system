import io
import cv2
import numpy as np

from insightface.app import FaceAnalysis

# Initialize face app once (CPU via onnxruntime)
# name='buffalo_l' provides detection+alignment+embedding pipeline
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

def read_image_from_bytes(data: bytes):
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image data")
    return img

def get_best_face(img):
    faces = face_app.get(img)
    if not faces:
        return None
    return max(faces, key=lambda f: getattr(f, "det_score", 0.0))

def extract_normed_embedding(img: np.ndarray):
    face = get_best_face(img)
    if face is None:
        return None, None
    emb = face.normed_embedding
    bbox = face.bbox.astype(int).tolist()
    return emb, bbox

def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.dot(u, v))

def best_match(query_emb: np.ndarray, embeddings: list[tuple[int, np.ndarray]]) -> tuple[int | None, float]:
    """
    embeddings: list of (student_id, emb_vector)
    Returns (best_student_id, best_similarity)
    """
    best_sid = None
    best_sim = -1.0
    for sid, emb in embeddings:
        sim = cosine_similarity(query_emb, emb)
        if sim > best_sim:
            best_sim = sim
            best_sid = sid
    return best_sid, best_sim
