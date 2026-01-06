import os
import uuid
import math
from typing import Optional, Union
from datetime import datetime

import numpy as np
import cv2
import mediapipe as mp

from datetime import datetime, timedelta
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from .video import extract_best_frame_from_video
from insightface.app import FaceAnalysis

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from .db import SessionLocal, engine, Base
from .models import Student, FaceEmbedding, FaceImage, Attendance, Exam, ExamSession, ExamIncident
from .config import FACE_THRESHOLD, SAVE_IMAGES, IMAGE_DIR
from .face import read_image_from_bytes, extract_normed_embedding, best_match
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)  # use ctx_id=1 if you have GPU


def save_image(student_id: int, img_bytes: bytes) -> str:
    fname = f"{student_id}_{uuid.uuid4().hex}.jpg"
    fpath = os.path.join(IMAGE_DIR, fname)
    with open(fpath, "wb") as f:
        f.write(img_bytes)
    return fpath


def load_all_embeddings(db: Session):
    rows = db.execute(select(FaceEmbedding.student_id, FaceEmbedding.embedding)).all()
    emb_list = []
    for sid, emb_json in rows:
        vec = np.array(emb_json, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        emb_list.append((sid, vec))
    return emb_list


def gen_frames(camera):
    while True:
        success, frame = camera.read()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def log_incident(db: Session, session_id: int, kind: str, details: str = None):
    db.add(ExamIncident(session_id=session_id, incident_type=kind, details=details or ""))
    db.commit()


def count_recent_incidents(db: Session, session_id: int, kind: str, seconds: int) -> int:
    rows = db.execute(text("""
        SELECT COUNT(*) AS c
        FROM exam_incidents
        WHERE session_id = :sid
          AND incident_type = :kind
          AND created_at >= (NOW() - INTERVAL :secs SECOND)
    """), {"sid": session_id, "kind": kind, "secs": seconds}).fetchone()
    return rows[0] if rows else 0


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

YAW_MAX_DEG = 30
PITCH_UP_MAX = 25
PITCH_DOWN_MIN = -30
LOOK_AWAY_SECONDS = 2.5

def estimate_head_pose_degrees_bounded(image_bgr):
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(img_rgb)
    if not res.multi_face_landmarks:
        return None

    h, w = image_bgr.shape[:2]
    lm = res.multi_face_landmarks[0].landmark

    idx = [1, 33, 263, 199, 61, 291]

    face_2d = []
    for i in idx:
        x = int(lm[i].x * w)
        y = int(lm[i].y * h)
        face_2d.append([x, y])

    face_2d = np.array(face_2d, dtype=np.float64)

    face_3d = np.array([
        [0.0, 0.0, 0.0],          # Nose
        [-225.0, -170.0, -135.0], # Left eye
        [225.0, -170.0, -135.0],  # Right eye
        [0.0, -330.0, -65.0],     # Chin
        [-150.0, 150.0, -125.0],  # Left mouth
        [150.0, 150.0, -125.0]    # Right mouth
    ])

    focal_length = w
    center = (w / 2, h / 2)
    cam_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist)
    if not success:
        return None

    rot_mat, _ = cv2.Rodrigues(rvec)

    pitch = math.degrees(math.atan2(-rot_mat[2, 1], rot_mat[2, 2]))
    yaw   = math.degrees(math.atan2(rot_mat[1, 0], rot_mat[0, 0]))

    return yaw, pitch


Base.metadata.create_all(bind=engine)
app = FastAPI(title="Face Recognition Attendance System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(IMAGE_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

camera = cv2.VideoCapture(0)


@app.post("/students/register")
async def register_student(
    name: str = Form(...),
    email: Optional[str] = Form(None),
    roll_number: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None)
):
    if video:
        video_bytes = await video.read()
        frame, emb, bbox = extract_best_frame_from_video(video_bytes)
        if emb is None:
            raise HTTPException(status_code=400, detail="No face detected in video.")
        img_bytes = None
    elif image:
        img_bytes = await image.read()
        frame = read_image_from_bytes(img_bytes)
        emb, bbox = extract_normed_embedding(frame)
        if emb is None:
            raise HTTPException(status_code=400, detail="No face detected.")
    else:
        raise HTTPException(status_code=400, detail="No image or video provided.")

    with SessionLocal() as db:
        student = Student(name=name, email=email, roll_number=roll_number)
        db.add(student)
        db.flush()

        if SAVE_IMAGES:
            if frame is not None:
                if img_bytes is None:
                    _, buffer = cv2.imencode('.jpg', frame)
                    img_bytes = buffer.tobytes()
                path = save_image(student.id, img_bytes)
                db.add(FaceImage(student_id=student.id, image_path=path))

        db.add(FaceEmbedding(student_id=student.id, embedding=emb.tolist()))
        db.commit()
        db.refresh(student)

    return {"status": "ok", "student_id": student.id, "bbox": bbox}


@app.post("/students/{student_id}/add-embedding")
async def add_embedding(
    student_id: int,
    image: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None)
):
    if video:
        video_bytes = await video.read()
        frame, emb, bbox = extract_best_frame_from_video(video_bytes)
        if emb is None:
            raise HTTPException(status_code=400, detail="No face detected in video.")
        img_bytes = None
    elif image:
        img_bytes = await image.read()
        frame = read_image_from_bytes(img_bytes)
        emb, bbox = extract_normed_embedding(frame)
        if emb is None:
            raise HTTPException(status_code=400, detail="No face detected.")
    else:
        raise HTTPException(status_code=400, detail="No image or video provided.")

    with SessionLocal() as db:
        st = db.get(Student, student_id)
        if not st:
            raise HTTPException(status_code=404, detail="Student not found")

        if SAVE_IMAGES and frame is not None:
            if img_bytes is None:
                _, buffer = cv2.imencode('.jpg', frame)
                img_bytes = buffer.tobytes()
            path = save_image(student_id, img_bytes)
            db.add(FaceImage(student_id=student_id, image_path=path))

        db.add(FaceEmbedding(student_id=student_id, embedding=emb.tolist()))
        db.commit()

    return {"status": "ok", "student_id": student_id, "bbox": bbox}
@app.post("/students/register_video")
async def register_student_video(
    name: str = Form(...),
    email: Optional[str] = Form(None),
    roll_number: Optional[str] = Form(None),
    video: UploadFile = File(...),
):
    video_bytes = await video.read()
    frame, emb, bbox = extract_best_frame_from_video(video_bytes)
    if emb is None:
        raise HTTPException(status_code=400, detail="No face detected in video.")

    with SessionLocal() as db:
        student = Student(name=name, email=email, roll_number=roll_number)
        db.add(student)
        db.flush() 
        if SAVE_IMAGES and frame is not None:
            _, buffer = cv2.imencode('.jpg', frame)
            path = save_image(student.id, buffer.tobytes())
            db.add(FaceImage(student_id=student.id, image_path=path))

        db.add(FaceEmbedding(student_id=student.id, embedding=emb.tolist()))
        db.commit()
        db.refresh(student)

    return {"status": "ok", "student_id": student.id, "bbox": bbox}

@app.post("/students/{student_id}/add-embedding_video")
async def add_embedding_video(student_id: int, video: UploadFile = File(...)):
    video_bytes = await video.read()
    frame, emb, bbox = extract_best_frame_from_video(video_bytes)
    if emb is None:
        raise HTTPException(status_code=400, detail="No face detected in video.")

    with SessionLocal() as db:
        student = db.get(Student, student_id)
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")

        if SAVE_IMAGES and frame is not None:
            _, buffer = cv2.imencode('.jpg', frame)
            path = save_image(student_id, buffer.tobytes())
            db.add(FaceImage(student_id=student_id, image_path=path))

        db.add(FaceEmbedding(student_id=student_id, embedding=emb.tolist()))
        db.commit()

    return {"status": "ok", "student_id": student_id, "bbox": bbox}



@app.post("/students/{student_id}/add-embedding")
async def add_embedding(student_id: int, image: UploadFile = File(...)):
    img_bytes = await image.read()
    img = read_image_from_bytes(img_bytes)
    emb, bbox = extract_normed_embedding(img)
    if emb is None:
        raise HTTPException(status_code=400, detail="No face detected.")

    with SessionLocal() as db:
        st = db.get(Student, student_id)
        if not st:
            raise HTTPException(status_code=404, detail="Student not found")
        if SAVE_IMAGES:
            path = save_image(student_id, img_bytes)
            db.add(FaceImage(student_id=student_id, image_path=path))
        db.add(FaceEmbedding(student_id=student_id, embedding=emb.tolist()))
        db.commit()

    return {"status": "ok", "student_id": student_id, "bbox": bbox}


@app.get("/students")
def get_all_students():
    with SessionLocal() as db:
        students = db.query(Student).all()
        return [
            {"id": s.id, "name": s.name, "email": s.email, "roll_number": s.roll_number}
            for s in students
        ]


@app.post("/attendance/mark")
async def mark_attendance(
    class_name: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    image: UploadFile = File(...),
):
    img_bytes = await image.read()
    img = read_image_from_bytes(img_bytes)
    emb, bbox = extract_normed_embedding(img)
    if emb is None:
        raise HTTPException(status_code=400, detail="No face detected.")

    with SessionLocal() as db:
        all_embs = load_all_embeddings(db)
        if not all_embs:
            raise HTTPException(status_code=404, detail="No registered students yet.")

        sid, sim = best_match(emb, all_embs)
        if sid is None or sim < FACE_THRESHOLD:
            return JSONResponse(status_code=403, content={
                "status": "rejected",
                "reason": "Unregistered face",
                "similarity": sim,
                "threshold": FACE_THRESHOLD,
                "bbox": bbox,
            })

        rec = Attendance(student_id=sid, class_name=class_name, session_id=session_id)
        db.add(rec)
        db.commit()

        student = db.get(Student, sid)
        return {
            "status": "allowed",
            "student_id": sid,
            "name": student.name if student else None,
            "similarity": sim,
            "threshold": FACE_THRESHOLD,
            "bbox": bbox,
        }


@app.get("/attendance/today")
def attendance_today():
    with SessionLocal() as db:
        rows = db.execute(text("""
            SELECT a.id, s.name, s.roll_number, a.class_name, a.session_id, a.attended_at
            FROM attendance a
            JOIN students s ON s.id = a.student_id
            WHERE DATE(a.attended_at) = CURDATE()
            ORDER BY a.attended_at DESC
        """)).mappings().all()
        return {"rows": [dict(r) for r in rows]}


@app.get("/images/{filename}")
def get_image(filename: str):
    path = os.path.join(IMAGE_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path)


@app.get("/webcam", response_class=HTMLResponse)
async def webcam_page(request: Request):
    return templates.TemplateResponse("webcam.html", {"request": request})


@app.post("/exam/start")
async def exam_start(student_id: int = Form(...), exam_code: str = Form(...), image: UploadFile = File(...)):
    img_bytes = await image.read()
    img = read_image_from_bytes(img_bytes)
    emb, _ = extract_normed_embedding(img)
    if emb is None:
        raise HTTPException(status_code=400, detail="No face detected")

    with SessionLocal() as db:
        exam = db.execute(select(Exam).where(Exam.exam_code == exam_code)).scalar_one_or_none()
        if not exam:
            exam = Exam(exam_code=exam_code, title=exam_code)
            db.add(exam)
            db.commit()
            db.refresh(exam)

        all_embs = load_all_embeddings(db)
        sid, sim = best_match(emb, all_embs)
        if sid != student_id or sim < FACE_THRESHOLD:
            raise HTTPException(status_code=403, detail="Face verification failed")

        session = ExamSession(exam_id=exam.id, student_id=student_id, status="active")
        db.add(session)
        db.commit()
        db.refresh(session)
        return {"status": "exam_started", "session_id": session.id, "similarity": sim, "threshold": FACE_THRESHOLD}


@app.post("/exam/monitor")
async def exam_monitor(session_id: int = Form(...), image: UploadFile = File(...)):
    img_bytes = await image.read()
    frame = read_image_from_bytes(img_bytes)

    with SessionLocal() as db:
        session = db.get(ExamSession, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        if session.status != "active":
            return {"status": session.status, "reason": session.reason_locked}

        faces = face_app.get(frame)
        if len(faces) != 1:
            session.status = "locked"
            session.reason_locked = "No or multiple faces detected"
            db.commit()
            log_incident(db, session.id, "face_count")
            return {"status": "locked"}

        now = datetime.utcnow()

        if session.look_away_start is not None and not isinstance(session.look_away_start, datetime):
            session.look_away_start = None
            db.commit()

        pose = estimate_head_pose_degrees_bounded(frame)

        if pose:
            yaw, pitch = pose
            yaw = max(min(yaw, 60.0), -60.0)
            pitch = max(min(pitch, 45.0), -45.0)

            looking_away = (
                abs(yaw) > YAW_MAX_DEG or
                pitch < PITCH_DOWN_MIN or
                pitch > PITCH_UP_MAX
            )

            if looking_away:
                if session.look_away_start is None:
                    session.look_away_start = now
                    db.commit()
                    log_incident(db, session.id, "looking_away_start")
                else:
                    elapsed = (now - session.look_away_start).total_seconds()
                    if elapsed >= LOOK_AWAY_SECONDS:
                        session.status = "locked"
                        session.reason_locked = "Looking away too long"
                        db.commit()
                        log_incident(db, session.id, "looking_away_locked")
                        return {"status": "locked"}
            else:
                if session.look_away_start:
                    elapsed = (now - session.look_away_start).total_seconds()
                    if elapsed > 0.3:
                        session.look_away_start = None
                        db.commit()

        emb, _ = extract_normed_embedding(frame)
        if emb is None:
            session.status = "locked"
            session.reason_locked = "Face unclear"
            db.commit()
            return {"status": "locked"}

        all_embs = load_all_embeddings(db)
        sid, sim = best_match(emb, all_embs)

        if sid != session.student_id or sim < FACE_THRESHOLD:
            session.status = "locked"
            session.reason_locked = "Face mismatch"
            db.commit()
            log_incident(db, session.id, "face_mismatch")
            return {"status": "locked"}

        return {"status": "active"}



@app.post("/exam/end")
async def exam_end(session_id: int = Form(...)):
    with SessionLocal() as db:
        session = db.get(ExamSession, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        session.status = "completed"
        session.end_time = datetime.utcnow()
        db.commit()
        return {"status": "completed"}


@app.get("/exam/session/{session_id}")
def exam_session_status(session_id: int):
    with SessionLocal() as db:
        session = db.get(ExamSession, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return {
            "session_id": session.id,
            "status": session.status,
            "reason_locked": session.reason_locked,
            "student_id": session.student_id,
            "exam_id": session.exam_id
        }

@app.get("/exam", response_class=HTMLResponse)
async def exam_page(request: Request):
    return templates.TemplateResponse("exam.html", {"request": request})

