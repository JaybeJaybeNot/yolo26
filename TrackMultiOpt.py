
from pathlib import Path
import time
from ultralytics import YOLO
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import math


# IMPORTANT:
# Color detection currently processes the same frame as YOLO,
# but its result is retrieved a few frames later due to asynchronous processing.
# This can be adjusted for a live camera implementation.

skip = 2

root = Path(__file__).resolve().parent

videos = [
    "[BLUR_TEST]RoboMasterTestVideo3.mp4",
    "RoboMasterTestVideo2.mp4",
]
#parameters for alpha beta filter
alpha = 0.85
beta = (alpha * alpha)/ (2- alpha)
threshold = 15
max_missing_frames = 15



class plate:
    def __init__(self, x, y, id, frame_id):
        self.x = float(x)
        self.y = float(y)
        self.dx = 0.0
        self.dy = 0.0
        self.id = id
        self.frame_id = frame_id
        self.last_yolo_frame = frame_id
        self.last_yolo_x = float(x)
        self.last_yolo_y = float(y)
        self.last_detection_frame = frame_id

    def move(self, frame_count):
        dt = frame_count - self.last_yolo_frame
        pred_x = self.x + self.dx * dt
        pred_y = self.y + self.dy * dt

        rx = self.last_yolo_x - pred_x
        ry = self.last_yolo_y - pred_y

        if dt > 0:
            self.dx += beta * rx / dt
            self.dy += beta * ry / dt

        self.x = pred_x + alpha * rx
        self.y = pred_y + alpha * ry
        self.frame_id = frame_count
        self.last_yolo_frame = frame_count

    def reset(self, x, y):
        self.x = x
        self.y = y


class robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

    color: str
    plates: plate


class Track:  # Regions of interest
    def __init__(self, x1, y1, x2, y2, margin=80):
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
    # Contains all Regions of Interest
]
#set contain plates
plates = {

}

next_plate_id = 0


def track_video(model, video_path, enemyColor, allyColor, maxPixDistance):
    global next_plate_id

    plates.clear()
    next_plate_id = 0

    video_path = Path(video_path)
    if not video_path.is_absolute():
        video_path = root / video_path
    frame_count = 0
    start_time = time.perf_counter()

    # One CPU worker for color detection and grouping
    executor = ThreadPoolExecutor(max_workers=1)

    # Limit outstanding jobs to prevent memory accumulation
    pending = deque()

    def remove_expired_plates(current_frame):
        expired_ids = [
            plate_id
            for plate_id, p in plates.items()
            if current_frame - p.last_detection_frame > max_missing_frames
        ]
        for plate_id in expired_ids:
            del plates[plate_id]

    results = model.predict(
        source=video_path,
        imgsz=256,
        #127 fps at 418, 0.9 ms inference
        #127 fps at 224, 0.9 ms inference
        conf=0.1,
        tracker=str(root / "byteTrackConfig.yaml"),
        save=False,
        device=0,
        stream=True,
        vid_stride = skip,
    )
    try:
        for result in results:




            frame_count += 1
            remove_expired_plates(frame_count)

            # Extract YOLO bounding boxes _______________OPT IMPLEMENTATION_______
            
            plate_boxes = result.boxes.xyxy.cpu().numpy()
            
            xyxy_list = result.boxes.xyxy.cpu().numpy().tolist()
            

            
            # 2. Update Plates = { } set from yolo results
            matched_ids = set()
            for xyxy in xyxy_list:
                xmin, ymin, xmax, ymax = xyxy
                center_x = (xmin + xmax) / 2
                center_y = (ymin + ymax) / 2

                matched = False

                for p in plates.values():
                    if p.id in matched_ids:
                        continue

                    distance = ((center_x - p.x)**2 + (center_y - p.y)**2)**0.5

                    if distance < maxPixDistance:
                        dt = frame_count - p.last_yolo_frame
                        if dt > 0:
                            pred_x = p.x + p.dx * dt
                            pred_y = p.y + p.dy * dt

                            p.dx += beta * (center_x - pred_x) / dt
                            p.dy += beta * (center_y - pred_y) / dt
                        p.x = center_x
                        p.y = center_y
                        p.last_yolo_frame = frame_count
                        p.last_detection_frame = frame_count
                        matched_ids.add(p.id)
                        matched = True
                        break


                #add new plate

                if not matched:
                    plates[next_plate_id] = plate(
                        center_x, center_y, next_plate_id, frame_count
                    )
                    matched_ids.add(next_plate_id)
                    next_plate_id += 1

            
            #predicting the skip-1 frames 
            for i in range(skip - 1):
                frame_count += 1
                for p in list(plates.values()):
                    p.move(frame_count)
                remove_expired_plates(frame_count)



                
#________________________________________
            # Access the SAME frame YOLO already decoded
            frame = result.orig_img

            # Submit grouping asynchronously
            future = executor.submit(
                grouping,
                frame, plate_boxes,enemyColor, allyColor, maxPixDistance
            )
            # Store frame number and future
            pending.append((frame_count, future))
            # Limit pending jobs to 2
            if len(pending) > 2:
                frame_id, completed = pending.popleft()
                matches = completed.result()
                # Process grouping results
                for team, center in matches:
                    print(f"Frame: {frame_id} | "
                        f"Team: {team} | "
                        f"Center: {center}"
                    )
        # Collect remaining jobs
        while pending:
            frame_id, completed = pending.popleft()
            matches = completed.result()
            for team, center in matches:
                print(
                    f"Frame: {frame_id} | "
                    f"Team: {team} | "
                    f"Center: {center}"
                )
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

    color_ranges = {


        "red": [
            (
                np.array([0, 100, 100]),
                np.array([10, 255, 255])
            ),
            (
                np.array([170, 100, 100]),
                np.array([180, 255, 255])
            )
        ],


        "blue": [
            (
                np.array([100, 100, 100]),
                np.array([130, 255, 255])
            )
        ]
    }


    detections = []

    for color, ranges in color_ranges.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(hsv, lower, upper)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            if cv2.contourArea(contour) > 100:
                x, y, w, h = cv2.boundingRect(contour)
                detections.append((color, (x, y, w, h)))

    return detections

#TBD have to optimize some goofy ahh ai code
def grouping(image, plate_boxes, enemyColor, allyColor, maxPixDistance):
    """
    Match YOLO bounding boxes to nearby color regions.

    Parameters:
        image: Original BGR frame.
        plate_boxes: YOLO bounding boxes, shape (N, 4).
        enemyColor: "red" or "blue".
        allyColor: "red" or "blue".
        maxPixDistance: Maximum distance between plate and color region.

    Returns:
        List of (team, center) tuples.

        Example:
            [
                ("enemy", (320, 240)),
                ("ally", (510, 180))
            ]
    """

    # ############__________________________________________________Validate inputs
    enemyColor = enemyColor.lower()
    allyColor = allyColor.lower()
    available_colors = {"red", "blue"}
    if enemyColor not in available_colors or allyColor not in available_colors:
        raise ValueError(
            "enemyColor and allyColor must be 'red' or 'blue'"
        )
    if enemyColor == allyColor:
        raise ValueError(
            "enemyColor and allyColor must be different"
        )
    if maxPixDistance < 0:
        raise ValueError(
            "maxPixDistance must be non-negative"
        )
    if image is None or plate_boxes is None or len(plate_boxes) == 0:
        return []
        #__________________________________________________

    # Detect color regions
    color_boxes = color_detection(image)

    if not color_boxes:
        return []

    # Keep only the relevant team colors
    color_boxes = [
        (color, box)
        for color, box in color_boxes
        if color in (enemyColor, allyColor)
    ]

    if not color_boxes:
        return []

    candidates = []

    # Compare every YOLO box against every color region
    for plate_index, plate_box in enumerate(plate_boxes):

        px1, py1, px2, py2 = plate_box

        # Calculate center of YOLO bounding box
        center = (
            int((px1 + px2) / 2),
            int((py1 + py2) / 2)
        )

        for region_index, (color, (rx, ry, rw, rh)) in enumerate(color_boxes):

            # Calculate distance between rectangles
            dx = max(px1 - (rx + rw), rx - px2, 0)
            dy = max(py1 - (ry + rh), ry - py2, 0)

            distance = (dx * dx + dy * dy) ** 0.5

            if distance <= maxPixDistance:

                team = "enemy" if color == enemyColor else "ally"

                candidates.append(
                    (distance, plate_index, region_index, team, center)
                )

    # Greedy nearest-neighbor matching
    matches = []

    matched_plates = set()
    matched_regions = set()

    for _, plate_index, region_index, team, center in sorted(candidates):

        if plate_index in matched_plates or region_index in matched_regions:
            continue

        matched_plates.add(plate_index)
        matched_regions.add(region_index)

        matches.append(
            (team, center)
        )

    return matches


def main():
    model_path = (
        root / "runs/detect/[LARGE]Robomasters.v6i.yolo26_EPOCH100/weights/best.engine"
    )
    model = YOLO(str(model_path))

    # Team color configuration
    enemyColor = "red"
    allyColor = "blue"
    # Maximum pixel distance for matching a plate to a color region
    maxPixDistance = 80

    for video in videos:

        track_video(
            model,
            video,
            enemyColor,
            allyColor,
            maxPixDistance
        )


if __name__ == "__main__":
    main()
