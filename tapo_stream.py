import os
import cv2
from urllib.parse import quote
from dotenv import load_dotenv

load_dotenv()

user = quote(os.environ["TAPO_USER"], safe="")
pw   = quote(os.environ["TAPO_PASS"], safe="")
host = os.environ["TAPO_HOST"]
port = os.getenv("TAPO_PORT", "554")

rtsp_url = f"rtsp://{user}:{pw}@{host}:{port}/stream1"

cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise RuntimeError("Konnte RTSP-Stream nicht öffnen")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame verloren, reconnecting…")
            cap.release()
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            continue
        cv2.imshow("Tapo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()