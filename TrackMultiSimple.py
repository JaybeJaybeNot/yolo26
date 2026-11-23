from pathlib import Path
import time
from ultralytics import YOLO
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from collections import deque


root = Path(__file__).resolve().parent
videos = [
    "[BLUR_TEST]RoboMasterTestVideo3.mp4",
    "RoboMasterTestVideo2.mp4",
]

# IMPORTANT:
# Color detection currently processes the same frame as YOLO,
# but its result is retrieved a few frames later due to asynchronous processing.
# This can be adjusted for a live camera implementation.


class plate:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def move(self, dx, dy):
        self.x += dx
        self. y += dy




class robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def move(self, dx, dy):
        self.x += dx
        self. y += dy
    color: str
    plates: plate


class Track: #Regions of interest
    def __init__(self, track_id, x1, y1, x2, y2, margin=80):
        self.id = track_id
        self.x = (x1 + x2) / 2
        self.y = (y1 + y2) / 2
        self.width = x2 - x1
        self.height = y2 - y1
        self.margin = margin
        self.missing_frames = 0


    def update(self, x1, y1, x2, y2):
        self.x = (x1 + x2) / 2
        self.y = (y1 + y2) / 2
        self.width = x2 - x1
        self.height = y2 - y1
        self.missing_frames = 0


    def get_roi(self, frame_width, frame_height):
        return (
            max(0, int(self.x - self.width / 2 - self.margin)),
            max(0, int(self.y - self.height / 2 - self.margin)),
            min(frame_width, int(self.x + self.width / 2 + self.margin)),
            min(frame_height, int(self.y + self.height / 2 + self.margin))
        )


tracks = [
    #contains all Regions of Interest
]


def track_video(model, video_path):
    video_path = Path(video_path)
    if not video_path.is_absolute():
        video_path = root / video_path

    frame_count = 0
    start_time = time.perf_counter()

    # One CPU worker for color detection
    executor = ThreadPoolExecutor(max_workers=1)
    # Limit outstanding jobs to prevent memory accumulation
    pending = deque()


    results = model.predict(
        source=video_path,
        imgsz=224,
        conf=0.1,
        tracker=str(root / "byteTrackConfig.yaml"),
        save=False,
        device=0,
        stream=True,
    )


    try:
        for result in results:


            frame_count += 1


            # Access the SAME frame YOLO already decoded
            frame = result.orig_img
            # Submit color detection asynchronously
            future = executor.submit( #Where multithreading occurs
                color_detection,
                frame
            )
            pending.append((frame_count, future))
            # Keep at most two outstanding color-processing jobs
            if len(pending) > 2:
                frame_id, completed = pending.popleft()
                color_results = completed.result()
                # Combine with tracking information here
                # print(frame_id, color_results)
        # Collect remaining jobs
        while pending:
            frame_id, completed = pending.popleft()
            color_results = completed.result()
            # Final tracking fusion
            # print(frame_id, color_results)


    finally:
        executor.shutdown(wait=True)


    elapsed = time.perf_counter() - start_time


    if elapsed > 0:
        print(
            f"Tracking speed for {video_path.name}: "
            f"{frame_count / elapsed:.2f} FPS "
            f"({frame_count} frames over {elapsed:.2f}s)"
        )
    return frame_count




def color_detection(frame):
    """
    Runs on CPU.
    Receives the original frame from YOLO.
    Does not modify the original frame.
    """

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Example: detect a particular color
    lower = np.array([0, 100, 100])
    upper = np.array([10, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []

    for contour in contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            detections.append((x, y, w, h))


    return detections






   
def main():
    model_path = (
        root / "runs/detect/[LARGE]Robomasters.v6i.yolo26_EPOCH100/weights/best.engine"
    )
    model = YOLO(str(model_path))
    for video in videos:
        track_video(model, video)


if __name__ == "__main__":
    main()



