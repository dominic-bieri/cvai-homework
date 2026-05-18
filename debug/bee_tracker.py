import os
import cv2
from ultralytics import YOLO
from dotenv import load_dotenv
from urllib.parse import quote

load_dotenv()
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

user = quote(os.environ["TAPO_USER"], safe="")
pw   = quote(os.environ["TAPO_PASS"], safe="")
host = os.environ["TAPO_HOST"]
port = os.getenv("TAPO_PORT", "554")
rtsp_url = f"rtsp://{user}:{pw}@{host}:{port}/stream1"

model = YOLO("runs/detect/local_bee_models/yolo26_run_01/weights/best.pt")
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

while True:
    ok, frame = cap.read()
    if not ok:
        cap.release()
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        continue

    results = model.track(frame, persist=True, verbose=False)
    annotated = results[0].plot(line_width=1, font_size=6)

    cv2.imshow("Bee Tracker", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
