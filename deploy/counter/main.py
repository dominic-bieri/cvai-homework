import os
import time
import sqlite3
import logging
import cv2
from pathlib import Path
from urllib.parse import quote
from datetime import datetime, timezone
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DB_PATH = Path("/data/bees.db")
SNAPSHOTS_DIR = Path("/data/snapshots")
MODEL_PATH = os.getenv("MODEL_PATH", "/model/best.pt")
INTERVAL = int(os.getenv("INTERVAL_SECONDS", "60"))


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_rtsp_url() -> str:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    user = quote(os.environ.get("TAPO_USER", ""), safe="")
    pw = quote(os.environ.get("TAPO_PASS", ""), safe="")
    host = os.environ.get("TAPO_HOST", "")
    port = os.getenv("TAPO_PORT", "554")
    return f"rtsp://{user}:{pw}@{host}:{port}/stream1"


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS counts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                count       INTEGER NOT NULL,
                image_path  TEXT
            )
        """)
        con.commit()
    logger.info("Database ready at %s", DB_PATH)


def save_count(timestamp: str, count: int, image_path: str | None) -> None:
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO counts (timestamp, count, image_path) VALUES (?, ?, ?)",
            (timestamp, count, image_path),
        )
        con.commit()


def capture_frame(rtsp_url: str):
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logger.error("Could not open RTSP stream")
        return None
    time.sleep(2)
    for _ in range(30):
        cap.grab()
    ret, frame = cap.retrieve()
    cap.release()
    return frame if ret else None


def main() -> None:
    init_db()
    rtsp_url = get_rtsp_url()

    logger.info("Loading model from %s", MODEL_PATH)
    model = YOLO(MODEL_PATH)
    logger.info("Model loaded. Starting bee counter (interval: %ds)", INTERVAL)

    while True:
        start = time.time()
        timestamp = now_utc()

        frame = capture_frame(rtsp_url)
        if frame is None:
            logger.error("Frame capture failed — skipping")
        else:
            results = model(frame, verbose=False)
            count = len(results[0].boxes)
            logger.info("Bees detected: %d", count)

            # Save annotated snapshot
            img_filename = f"bee_{timestamp.replace(':', '-').replace('Z', '')}.jpg"
            img_path = SNAPSHOTS_DIR / img_filename
            annotated = results[0].plot(line_width=1, font_size=6)
            cv2.imwrite(str(img_path), annotated)

            # Keep only last 200 snapshots to save disk space
            snapshots = sorted(
                SNAPSHOTS_DIR.glob("*.jpg"), key=lambda p: p.stat().st_mtime
            )
            for old in snapshots[:-200]:
                old.unlink()

            save_count(timestamp, count, str(img_path))

        elapsed = time.time() - start
        time.sleep(max(0, INTERVAL - elapsed))


if __name__ == "__main__":
    main()
