import os
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "class_attendance")

FACE_THRESHOLD = float(os.getenv("FACE_THRESHOLD", "0.38"))
SAVE_IMAGES = os.getenv("SAVE_IMAGES", "true").lower() == "true"
IMAGE_DIR = os.getenv("IMAGE_DIR", "static/face_images")

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
