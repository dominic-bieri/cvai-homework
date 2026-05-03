import os
import time
import cv2
import logging
from ultralytics import YOLO
from dotenv import load_dotenv
from urllib.parse import quote

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def get_rtsp_url():
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    user = quote(os.environ.get("TAPO_USER", ""), safe="")
    pw   = quote(os.environ.get("TAPO_PASS", ""), safe="")
    host = os.environ.get("TAPO_HOST", "")
    port = os.getenv("TAPO_PORT", "554")
    return f"rtsp://{user}:{pw}@{host}:{port}/stream1"

def main():
    # Load YOLO Model
    trained_model = "runs/detect/local_bee_models/yolo26_run_01/weights/best.pt"

    logger.info(f"Loading model: {trained_model}")
    model = YOLO(trained_model)
    
    rtsp_url = get_rtsp_url()
    interval_seconds = 60

    logger.info(f"Starting bee counter. Interval: {interval_seconds}s")
    logger.info("Press Ctrl+C to stop.")

    try:
        while True:
            start_run = time.time()

            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                logger.error("Could not open RTSP stream. Retrying in next interval.")
            else:
                # Wait a moment and skip frames to clear buffer and reach a clean keyframe
                time.sleep(2) 
                for _ in range(30):
                    cap.grab()
                
                ret, frame = cap.retrieve()
                cap.release()

                if ret:
                    results = model(frame, verbose=False)
                    
                    # class 0 is 'bee'
                    count = len(results[0].boxes)
                    
                    logger.info(f"Bees detected: {count}")
                    
                    # Optional: save image for verification
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    results[0].save(filename=f"detections/bee_{timestamp}.jpg")
                else:
                    logger.error("Failed to retrieve frame from stream.")

            # Calculate sleep time to maintain interval
            elapsed = time.time() - start_run
            sleep_time = max(0, interval_seconds - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Stopping bee counter...")

if __name__ == "__main__":
    main()
