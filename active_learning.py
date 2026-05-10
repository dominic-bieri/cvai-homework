"""
Active learning pipeline: YOLO inference → Label Studio pre-annotation.

For each camera frame:
  1. Run YOLO detection.
  2. Upload the raw (un-annotated) frame via Label Studio's multipart import API.
     This creates a proper FileUpload record so Label Studio can serve the image.
  3. Convert bounding boxes to Label Studio's percentage-based format.
  4. POST the prediction to the created task via /api/predictions.

Standalone usage (captures one frame from the RTSP stream):
    python active_learning.py

Import for use in another script:
    from active_learning import push_frame_to_label_studio
    push_frame_to_label_studio(frame, model)
"""

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

# ── Label Studio config ───────────────────────────────────────────────────────
LS_URL           = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
LS_REFRESH_TOKEN = os.getenv("LABEL_STUDIO_REFRESH_TOKEN")
LS_PROJECT_ID    = int(os.getenv("LABEL_STUDIO_PROJECT_ID", "1"))

# ── Collection config ────────────────────────────────────────────────────────
DURATION_SECONDS = 60 * 60 * 5  # total run time
INTERVAL_SECONDS = 60 * 5   # gap between captures

# ── Model config ──────────────────────────────────────────────────────────────
MODEL_PATH    = "runs/detect/local_bee_models/yolo26_run_01/weights/best.pt"
MODEL_VERSION = "yolo26_run_01"
CLASS_NAMES   = ["Bee"]

# Must match your Label Studio project's label config XML:
#   <Image name="image" …/>            → TO_NAME   = "image"
#   <RectangleLabels name="label" …/>  → FROM_NAME = "label"
FROM_NAME = "label"
TO_NAME   = "image"


# ── JWT auth (Label Studio ≥ 1.15 disables legacy token auth) ─────────────────

_token_cache: dict = {"access": None, "exp": 0.0}


def _refresh_access_token() -> str:
    """Exchange the long-lived refresh token for a short-lived access token."""
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
    return access


def _access_token() -> str:
    """Return a valid access token, refreshing automatically when near expiry."""
    if not _token_cache["access"] or time.time() > _token_cache["exp"] - 60:
        _refresh_access_token()
    return _token_cache["access"]


def _json_headers() -> dict:
    return {"Authorization": f"Bearer {_access_token()}", "Content-Type": "application/json"}

def _auth_header() -> dict:
    return {"Authorization": f"Bearer {_access_token()}"}


# ── Label Studio API calls ────────────────────────────────────────────────────

def _upload_frame_as_task(frame) -> int:
    """
    Upload the raw frame as a JPEG via Label Studio's multipart import endpoint.
    This registers the file in Label Studio's database so it can be served correctly.

    Multipart file imports do NOT return task_ids in the response — only
    file_upload_ids. We look up the FileUpload record to get the exact
    UUID-prefixed image URL, then find the matching task from recent tasks.

    Returns the new task ID.
    """
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("cv2.imencode failed")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename  = f"bee_{timestamp}.jpg"

    resp = requests.post(
        f"{LS_URL}/api/projects/{LS_PROJECT_ID}/import",
        files={"file": (filename, io.BytesIO(buf.tobytes()), "image/jpeg")},
        headers=_auth_header(),   # no Content-Type — requests sets it for multipart
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    fu_ids = data.get("file_upload_ids", [])
    if not fu_ids:
        raise RuntimeError(f"No file_upload_ids in import response: {data}")

    # GET /api/import/file-upload/{id} → {"file": "upload/1/{uuid}-{name}.jpg", ...}
    fu_resp = requests.get(f"{LS_URL}/api/import/file-upload/{fu_ids[0]}", headers=_auth_header())
    fu_resp.raise_for_status()
    image_url = f"/data/{fu_resp.json()['file']}"   # e.g. /data/upload/1/abc123-bee.jpg

    # Find the task that was just created for this image URL
    tasks_resp = requests.get(
        f"{LS_URL}/api/projects/{LS_PROJECT_ID}/tasks?page_size=10&ordering=-id",
        headers=_auth_header(),
    )
    tasks_resp.raise_for_status()
    raw = tasks_resp.json()
    tasks = raw if isinstance(raw, list) else raw.get("tasks", raw.get("results", []))

    for task in tasks:
        if task.get("data", {}).get("image") == image_url:
            logger.info("Frame uploaded → task ID %d (%s)", task["id"], image_url)
            return task["id"]

    raise RuntimeError(f"Could not find task with image URL: {image_url}")


def _create_prediction(task_id: int, ls_results: list, score: float) -> requests.Response:
    """Attach a prediction (pre-annotation) to an existing Label Studio task."""
    payload = {
        "task":          task_id,
        "model_version": MODEL_VERSION,
        "score":         score,
        "result":        ls_results,
    }
    resp = requests.post(
        f"{LS_URL}/api/predictions",
        json=payload,
        headers=_json_headers(),
        timeout=30,
    )
    resp.raise_for_status()
    logger.info("Prediction attached to task %d (%d regions)", task_id, len(ls_results))
    return resp


# ── YOLO → Label Studio conversion ───────────────────────────────────────────

def _xyxy_to_ls_percent(x1, y1, x2, y2, img_w, img_h) -> dict:
    """Convert absolute pixel box [x1,y1,x2,y2] to Label Studio percentage coords."""
    return {
        "x":      x1 / img_w * 100,
        "y":      y1 / img_h * 100,
        "width":  (x2 - x1) / img_w * 100,
        "height": (y2 - y1) / img_h * 100,
        "rotation": 0,
    }


def _build_ls_results(det_results, img_w: int, img_h: int) -> tuple:
    """
    Convert a YOLO Results object into a list of Label Studio region dicts
    and an overall confidence score.
    """
    boxes      = det_results.boxes
    ls_results = []

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        conf    = float(boxes.conf[i])
        cls_idx = int(boxes.cls[i])
        label   = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else str(cls_idx)

        ls_results.append({
            "id":              uuid.uuid4().hex[:10],
            "type":            "rectanglelabels",
            "from_name":       FROM_NAME,
            "to_name":         TO_NAME,
            "original_width":  img_w,
            "original_height": img_h,
            "image_rotation":  0,
            "value": {
                **_xyxy_to_ls_percent(x1, y1, x2, y2, img_w, img_h),
                "rectanglelabels": [label],
                "score": conf,
            },
        })

    overall_score = float(boxes.conf.mean()) if len(boxes) > 0 else 0.0
    return ls_results, overall_score


# ── Public API ────────────────────────────────────────────────────────────────

def push_frame_to_label_studio(frame, model) -> int:
    """
    Run YOLO inference on `frame`, upload it to Label Studio, and attach
    pre-annotations to the created task.

    Args:
        frame:  BGR numpy array (e.g. from cv2.VideoCapture.retrieve()).
        model:  Loaded ultralytics YOLO instance.

    Returns:
        The Label Studio task ID that was created.

    Raises:
        RuntimeError: if LABEL_STUDIO_REFRESH_TOKEN is missing or upload fails.
        requests.HTTPError: if any API call fails.
    """
    img_h, img_w = frame.shape[:2]

    results     = model(frame, verbose=False)
    det_results = results[0]
    logger.info("Bees detected: %d", len(det_results.boxes))

    # Upload raw frame → creates task with a proper FileUpload record
    task_id = _upload_frame_as_task(frame)

    # Attach pre-annotations (skip if nothing was detected)
    ls_results, avg_score = _build_ls_results(det_results, img_w, img_h)
    if ls_results:
        _create_prediction(task_id, ls_results, avg_score)

    return task_id


# ── Standalone entry-point ────────────────────────────────────────────────────

def _get_rtsp_url() -> str:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    user = quote(os.environ.get("TAPO_USER", ""), safe="")
    pw   = quote(os.environ.get("TAPO_PASS", ""), safe="")
    host = os.environ.get("TAPO_HOST", "")
    port = os.getenv("TAPO_PORT", "554")
    return f"rtsp://{user}:{pw}@{host}:{port}/stream1"


def _capture_frame(rtsp_url: str):
    """Open the RTSP stream, flush stale frames, and return one clean frame."""
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
    """
    Capture a frame every `interval_seconds` for `duration_seconds` total,
    uploading each with YOLO pre-annotations to Label Studio.

    Args:
        duration_seconds:  How long to run in total.
        interval_seconds:  Gap between captures (wall-clock, including upload time).
    """
    model    = YOLO(MODEL_PATH)
    rtsp_url = _get_rtsp_url()

    end_time   = time.time() + duration_seconds
    uploaded   = 0
    cycle      = 0

    logger.info(
        "Collection started — duration: %ds, interval: %ds, ~%d frames expected",
        duration_seconds, interval_seconds, duration_seconds // interval_seconds,
    )

    try:
        while time.time() < end_time:
            cycle += 1
            cycle_start = time.time()
            remaining   = end_time - cycle_start
            logger.info("Cycle %d | %.0fs remaining", cycle, remaining)

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

            elapsed    = time.time() - cycle_start
            sleep_time = max(0.0, interval_seconds - elapsed)

            if time.time() + sleep_time >= end_time:
                break   # no point sleeping past the deadline

            logger.info("Next capture in %.0fs", sleep_time)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")

    logger.info("Collection finished — %d/%d frames uploaded.", uploaded, cycle)


def main() -> None:
    collect(DURATION_SECONDS, INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
