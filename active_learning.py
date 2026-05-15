# ==============================================================================
# HINWEIS: Dieses Skript wurde nicht vollständig selbst geschrieben.
# Viele Teile wurden kopiert oder mit KI generiert, da die Label Studio API
# auf dem ersten, einfachen Weg nicht wie gewünscht funktioniert hat.
# ==============================================================================

import io
import os
import time
import uuid
import base64
import json
import logging
import requests
import cv2
from ultralytics import YOLO
from dotenv import load_dotenv
from urllib.parse import quote

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LS_URL           = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
LS_REFRESH_TOKEN = os.getenv("LABEL_STUDIO_REFRESH_TOKEN")
LS_PROJECT_ID    = int(os.getenv("LABEL_STUDIO_PROJECT_ID", "1"))

DURATION_SECONDS = 60 * 60 * 5
INTERVAL_SECONDS = 60 * 5

MODEL_PATH    = "runs/detect/local_bee_models/yolo26_run_01/weights/best.pt"
MODEL_VERSION = "yolo26_run_01"
CLASS_NAMES   = ["Bee"]

FROM_NAME = "label"
TO_NAME   = "image"

_token_cache: dict = {"access": None, "exp": 0.0}


def _headers(content_type: bool = False) -> dict:
    if not _token_cache["access"] or time.time() > _token_cache["exp"] - 60:
        if not LS_REFRESH_TOKEN:
            raise RuntimeError(
                "LABEL_STUDIO_REFRESH_TOKEN is not set.\n"
                "Add it to your .env file (Label Studio → Account & Settings → Access Token)."
            )
        resp = requests.post(
            f"{LS_URL}/api/token/refresh",
            json={"refresh": LS_REFRESH_TOKEN},
            timeout=10,
        )
        resp.raise_for_status()
        access = resp.json()["access"]
        padding = "=" * (4 - len(access.split(".")[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(access.split(".")[1] + padding))
        _token_cache["access"] = access
        _token_cache["exp"] = float(payload["exp"])

    h = {"Authorization": f"Bearer {_token_cache['access']}"}
    if content_type:
        h["Content-Type"] = "application/json"
    return h


def _upload_frame_as_task(frame) -> int:
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("cv2.imencode failed")

    filename = f"bee_{time.strftime('%Y%m%d-%H%M%S')}.jpg"

    resp = requests.post(
        f"{LS_URL}/api/projects/{LS_PROJECT_ID}/import",
        files={"file": (filename, io.BytesIO(buf.tobytes()), "image/jpeg")},
        headers=_headers(),
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    fu_ids = data.get("file_upload_ids", [])
    if not fu_ids:
        raise RuntimeError(f"No file_upload_ids in import response: {data}")

    fu_resp = requests.get(f"{LS_URL}/api/import/file-upload/{fu_ids[0]}", headers=_headers())
    fu_resp.raise_for_status()
    image_url = f"/data/{fu_resp.json()['file']}"

    tasks_resp = requests.get(
        f"{LS_URL}/api/projects/{LS_PROJECT_ID}/tasks?page_size=10&ordering=-id",
        headers=_headers(),
    )
    tasks_resp.raise_for_status()
    raw = tasks_resp.json()
    tasks = raw if isinstance(raw, list) else raw.get("tasks", raw.get("results", []))

    for task in tasks:
        if task.get("data", {}).get("image") == image_url:
            logger.info("Frame uploaded → task ID %d (%s)", task["id"], image_url)
            return task["id"]

    raise RuntimeError(f"Could not find task with image URL: {image_url}")


def _create_prediction(task_id: int, ls_results: list, score: float) -> None:
    resp = requests.post(
        f"{LS_URL}/api/predictions",
        json={"task": task_id, "model_version": MODEL_VERSION, "score": score, "result": ls_results},
        headers=_headers(content_type=True),
        timeout=30,
    )
    resp.raise_for_status()
    logger.info("Prediction attached to task %d (%d regions)", task_id, len(ls_results))


def _build_ls_results(det_results, img_w: int, img_h: int) -> tuple:
    boxes = det_results.boxes
    ls_results = [
        {
            "id":              uuid.uuid4().hex[:10],
            "type":            "rectanglelabels",
            "from_name":       FROM_NAME,
            "to_name":         TO_NAME,
            "original_width":  img_w,
            "original_height": img_h,
            "image_rotation":  0,
            "value": {
                "x":                boxes.xyxy[i][0].item() / img_w * 100,
                "y":                boxes.xyxy[i][1].item() / img_h * 100,
                "width":            (boxes.xyxy[i][2].item() - boxes.xyxy[i][0].item()) / img_w * 100,
                "height":           (boxes.xyxy[i][3].item() - boxes.xyxy[i][1].item()) / img_h * 100,
                "rotation":         0,
                "rectanglelabels":  [CLASS_NAMES[int(boxes.cls[i])] if int(boxes.cls[i]) < len(CLASS_NAMES) else str(int(boxes.cls[i]))],
                "score":            float(boxes.conf[i]),
            },
        }
        for i in range(len(boxes))
    ]
    overall_score = float(boxes.conf.mean()) if len(boxes) > 0 else 0.0
    return ls_results, overall_score


def push_frame_to_label_studio(frame, model) -> int:
    img_h, img_w = frame.shape[:2]

    det_results = model(frame, verbose=False)[0]
    logger.info("Bees detected: %d", len(det_results.boxes))

    task_id = _upload_frame_as_task(frame)

    ls_results, avg_score = _build_ls_results(det_results, img_w, img_h)
    if ls_results:
        _create_prediction(task_id, ls_results, avg_score)

    return task_id


def _capture_frame(rtsp_url: str):
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return None
    time.sleep(2)
    for _ in range(30):
        cap.grab()
    ret, frame = cap.retrieve()
    cap.release()
    return frame if ret else None


def collect(duration_seconds: int, interval_seconds: int) -> None:
    model    = YOLO(MODEL_PATH)
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    user     = quote(os.environ.get("TAPO_USER", ""), safe="")
    pw       = quote(os.environ.get("TAPO_PASS", ""), safe="")
    host     = os.environ.get("TAPO_HOST", "")
    port     = os.getenv("TAPO_PORT", "554")
    rtsp_url = f"rtsp://{user}:{pw}@{host}:{port}/stream1"

    end_time = time.time() + duration_seconds
    uploaded = 0
    cycle    = 0

    logger.info(
        "Collection started — duration: %ds, interval: %ds, ~%d frames expected",
        duration_seconds, interval_seconds, duration_seconds // interval_seconds,
    )

    try:
        while time.time() < end_time:
            cycle += 1
            cycle_start = time.time()
            logger.info("Cycle %d | %.0fs remaining", cycle, end_time - cycle_start)

            frame = _capture_frame(rtsp_url)
            if frame is None:
                logger.error("Could not capture frame — skipping cycle")
            else:
                try:
                    task_id = push_frame_to_label_studio(frame, model)
                    uploaded += 1
                    logger.info("Uploaded → task ID %d (total: %d)", task_id, uploaded)
                except Exception as exc:
                    logger.error("Upload failed: %s", exc)

            sleep_time = max(0.0, interval_seconds - (time.time() - cycle_start))
            if time.time() + sleep_time >= end_time:
                break

            logger.info("Next capture in %.0fs", sleep_time)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")

    logger.info("Collection finished — %d/%d frames uploaded.", uploaded, cycle)


def main() -> None:
    collect(DURATION_SECONDS, INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
