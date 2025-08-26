# Smart Classroom System (FastAPI + InsightFace + MediaPipe + MySQL)
This project provides a smart classroom backend with two main features:
- Face Recognition Attendance System
- Exam Monitoring (Cheating Detection)
It is designed for extensibility and easy deployment with FastAPI, MySQL, and lightweight ML models.

## Features

1. Attendance System
- Uses InsightFace (ONNX model) to generate face embeddings.
- Stores embeddings in MySQL for each student (no retraining needed).
- Marks attendance by matching live images against stored embeddings.
- Supports multiple photos per student for better accuracy.
- Threshold-based similarity check to reduce false matches.

2. Exam Monitoring System
- Uses MediaPipe (Google’s ML library) for real-time face/pose/landmark detection.
- Tracks:
    Face direction (looking away from screen)
    Multiple faces in frame
    Suspicious movement patterns
- Stores monitoring events in MySQL for later review.

## Tech Stack
- Backend: FastAPI (Python)
- ML Models:
    InsightFace (ONNX + ONNXRuntime) → Face embeddings
    MediaPipe → Face/pose landmark detection
- Database: MySQL (via phpMyAdmin/XAMPP for dev)
- Frontend (demo): Static HTML page with webcam capture
- Environment: Python venv

## Database design
Option 1: Import the SQL file  
```bash
mysql -u root -p class_attendance < database/class_attendance.sql
```

Option 2: Run the following queries manually in phpMyAdmin
```bash
-- Create database
CREATE DATABASE class_attendance;
USE class_attendance;

-- Attendance table
CREATE TABLE `attendance` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `student_id` int(11) NOT NULL,
  `attended_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `class_name` varchar(128) DEFAULT NULL,
  `session_id` varchar(64) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Exam results table
CREATE TABLE `exam_results` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `student_id` int(11) NOT NULL,
  `subject` varchar(128) DEFAULT NULL,
  `score` int(11) DEFAULT NULL,
  `exam_date` date DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Students table
CREATE TABLE `students` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  `email` varchar(255) DEFAULT NULL,
  `class` varchar(128) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

## Setup Instructions
1. Clone & Setup
```bash
git clone <your-repo>
cd smart-classroom
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
cp .env.example .env     # update with MySQL creds

```

2. Configure Database
- Open phpMyAdmin
- create database and tables using the query above
- Update .env file with your MySQL credentials.

3. Run server
```bash
uvicorn app.main:app --reload
```
Server runs at: http://127.0.0.1:8000
- Swagger UI: http://127.0.0.1:8000/docs
- Face attendance: http://127.0.0.1:8000/webcam
- Exam monitoring: http://127.0.0.1:8000/exam

## Notes
- Always run inside .venv:

```bash
pip install -r requirements.txt
```
- Good lighting + frontal faces improve accuracy
- Use HTTPS + auth in production.