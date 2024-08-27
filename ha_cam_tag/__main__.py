import logging
import re
import signal
import sys
import threading
import time
import json
import os
import requests
from datetime import datetime, timedelta
import cv2

# Constants
CONFIG_PATH = "/data/options.json"
API_URL = "http://supervisor/core/api/"
AUTH_TOKEN = os.environ['SUPERVISOR_TOKEN']
DEBOUNCE_PERIOD = timedelta(seconds=5)
TAG_ID_PATTERN = re.compile(r'https://www.home-assistant.io/tag/([0-9a-fA-F-]+)')

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')

def send_tag_event(tag_id, device_id):
    """Send a tag scanned event to Home Assistant."""
    endpoint = f"{API_URL}events/tag_scanned"
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}"
    }
    data = {
        "tag_id": tag_id,
        "device_id": device_id
    }
    try:
        response = requests.post(endpoint, headers=headers, json=data)
        response.raise_for_status()
        logging.info(f"Successfully sent tag event for {tag_id} from device {device_id}.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to send tag event: {e}")

def load_config():
    """Load configuration from the specified CONFIG_PATH."""
    try:
        with open(CONFIG_PATH, 'r') as fh:
            return json.load(fh)
    except FileNotFoundError:
        logging.error("Configuration file not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON config: {e}")
        sys.exit(1)

def detector_loop(cv, exiting, config):
    """Thread loop for detecting QR codes and sending tag events."""
    detector = cv2.QRCodeDetector()
    last_time, last_tag = None, None

    while not exiting.is_set():
        cv.wait()
        if frame is not None:
            try:
                data, _, _ = detector.detectAndDecode(frame)
                if data and (m := TAG_ID_PATTERN.match(data)):
                    cur_time, tag_id = datetime.now(), m.group(1)
                    if last_tag != tag_id or last_time < cur_time - DEBOUNCE_PERIOD:
                        send_tag_event(tag_id, config['tag_event_device_id'])
                        last_time, last_tag = cur_time, tag_id
            except cv2.error as e:
                logging.error(f"OpenCV error: {e}")
            except Exception as e:
                logging.exception(f"Unexpected error: {e}")
        else:
            logging.debug("No frame to process.")
        cv.clear()  # Clear the event for the next iteration

def main():
    config = load_config()
    exiting = threading.Event()
    frame = None
    cv = threading.Event()

    # Start detector thread
    detector_thread = threading.Thread(target=detector_loop, args=(cv, exiting, config))
    detector_thread.start()

    # Handle SIGINT
    signal.signal(signal.SIGINT, lambda sig, frame: exiting.set())

    try:
        while not exiting.is_set():
            stream = cv2.VideoCapture(config['camera_rtsp_stream'])
            if not stream.isOpened():
                logging.error("Failed to open camera stream.")
                time.sleep(5)
                continue

            while stream.isOpened() and not exiting.is_set():
                ret, frame = stream.read()
                if ret and frame is not None:
                    cv.set()  # Signal new frame is ready
                else:
                    logging.error("Failed to capture frame or frame is None.")
                    break  # Exit loop to restart the stream

            stream.release()
            time.sleep(5)

    except KeyboardInterrupt:
        logging.info("Exiting due to keyboard interrupt.")
        exiting.set()
        cv.set()
    finally:
        stream.release()
        cv2.destroyAllWindows()

    detector_thread.join()
    logging.info("Exited gracefully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())