import os
import logging
import cv2
from pathlib import Path
from ultralytics import YOLO
from dotenv import load_dotenv
from urllib.parse import quote

load_dotenv()
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

ZONE = (800, 700, 1450, 1000)  # (x1, y1, x2, y2) hive entrance
TRACKER = str(Path(__file__).resolve().parent / "bytetrack_bees.yaml")

user = quote(os.environ["TAPO_USER"], safe="")
pw = quote(os.environ["TAPO_PASS"], safe="")
host = os.environ["TAPO_HOST"]
port = os.getenv("TAPO_PORT", "554")
rtsp_url = f"rtsp://{user}:{pw}@{host}:{port}/stream1"

model = YOLO(
    Path(__file__).resolve().parent.parent.parent / "runs/detect/local_bee_models/yolo26_run_01/weights/best.pt")
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

prev_in_zone: dict[int, bool] = {}
entries = 0
exits = 0


def _in_zone(cx: float, cy: float) -> bool:
    x1, y1, x2, y2 = ZONE
    return x1 <= cx <= x2 and y1 <= cy <= y2


while True:
    ok, frame = cap.read()
    if not ok:
        cap.release()
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        continue

    results = model.track(
        frame,
        persist=True,
        verbose=False,
        conf=0.25,
        iou=0.45,
        tracker="bee_tracker.yaml"
    )
    annotated = results[0].plot(line_width=1, font_size=6)

    boxes = results[0].boxes
    if boxes is not None and boxes.id is not None:
        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()

        for bee_id, (x1, y1, x2, y2) in zip(ids, xyxy):
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            curr = _in_zone(cx, cy)

            if bee_id in prev_in_zone:
                if not prev_in_zone[bee_id] and curr:
                    entries += 1
                    logger.info("Einflug  bee_id=%d  (total: %d)", bee_id, entries)
                elif prev_in_zone[bee_id] and not curr:
                    exits += 1
                    logger.info("Ausflug  bee_id=%d  (total: %d)", bee_id, exits)

            prev_in_zone[bee_id] = curr

        prev_in_zone = {k: v for k, v in prev_in_zone.items() if k in set(ids)}

    zx1, zy1, zx2, zy2 = ZONE
    overlay = annotated.copy()
    cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), (0, 0, 200), -1)
    cv2.addWeighted(overlay, 0.20, annotated, 0.80, 0, annotated)
    cv2.rectangle(annotated, (zx1, zy1), (zx2, zy2), (0, 0, 255), 2)
    cv2.putText(annotated, "Eingang", (zx1 + 4, zy1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(annotated, f"Einflug:  {entries}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)
    cv2.putText(annotated, f"Ausflug: {exits}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 255), 2)

    cv2.imshow("Bee Zone Tracker", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
logger.info("Session beendet: %d Einflüge, %d Ausflüge", entries, exits)
