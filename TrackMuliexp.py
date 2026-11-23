from pathlib import Path
import time
from ultralytics import YOLO
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from collections import deque


# ============================================================
# CONFIGURATION
# ============================================================

root = Path(__file__).resolve().parent

videos = [
    "[BLUR_TEST]RoboMasterTestVideo3.mp4",
    "RoboMasterTestVideo2.mp4",
]

IMGSZ = 418 + 32 * 5
CONF = 0.1
DEVICE = 0

# Maximum distance allowed when associating plate centers.
MAX_TRACK_DISTANCE = 150


# Number of frames to skip after successful association.
SKIP_FRAMES = 2

# Maximum outstanding asynchronous color-grouping jobs.
MAX_PENDING = 2

MIN_MATCH_RATIO = 0.70


# ============================================================
# PLATE / ROBOT CLASSES
# ============================================================

class plate:
    def __init__(self, x, y, id):
        self.x = x
        self.y = y
        self.id = id

    def move(self, dx, dy):
        self.x += dx
        self.y += dy


class robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.color = None
        self.plates = []

    def move(self, dx, dy):
        self.x += dx
        self.y += dy


# ============================================================
# ROI TRACK CLASS
# ============================================================

class Track:
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
            min(frame_height, int(self.y + self.height / 2 + self.margin)),
        )


tracks = []
plates = {}


# ============================================================
# PLATE CENTER EXTRACTION
# ============================================================

def get_centers(boxes):
    """
    Convert bounding boxes (N, 4) into centers (N, 2).

    Input:
        [x1, y1, x2, y2]

    Output:
        [center_x, center_y]
    """

    if len(boxes) == 0:
        return np.empty((0, 2), dtype=np.float32)

    return np.column_stack((
        (boxes[:, 0] + boxes[:, 2]) / 2,
        (boxes[:, 1] + boxes[:, 3]) / 2
    ))


# ============================================================
# MATCH PLATES BETWEEN FRAMES
# ============================================================

def match_plates(prev_boxes, curr_boxes, max_distance=150):
    if len(prev_boxes) == 0 or len(curr_boxes) == 0:
        return None

    prev_centers = get_centers(prev_boxes)
    curr_centers = get_centers(curr_boxes)

    distances = np.linalg.norm(
        curr_centers[:, None, :] -
        prev_centers[None, :, :],
        axis=2
    )

    matched_indices = np.full(
        len(curr_boxes), -1, dtype=np.int32
    )

    used_prev = set()
    used_curr = set()

    for flat_index in np.argsort(distances, axis=None):
        curr_idx, prev_idx = np.unravel_index(
            flat_index, distances.shape
        )

        if curr_idx in used_curr or prev_idx in used_prev:
            continue

        if distances[curr_idx, prev_idx] > max_distance:
            break

        matched_indices[curr_idx] = prev_idx
        used_curr.add(curr_idx)
        used_prev.add(prev_idx)

    match_ratio = len(used_curr) / max(
        len(prev_boxes), len(curr_boxes)
    )

    if match_ratio < MIN_MATCH_RATIO:
        return None

    return matched_indices


# ============================================================
# CALCULATE PLATE VELOCITIES
# ============================================================

def calcChange(prev_boxes, curr_boxes, matched_indices, dt=1):
    prev_centers = get_centers(prev_boxes)
    curr_centers = get_centers(curr_boxes)

    velocities = np.zeros_like(curr_centers)

    for i, prev_idx in enumerate(matched_indices):
        if prev_idx == -1:
            continue

        velocities[i] = (
            curr_centers[i] - prev_centers[prev_idx]
        ) / dt

    return velocities


# ============================================================
# PREDICT PLATE LOCATIONS
# ============================================================

def predict_boxes(boxes, velocities, steps=1):
    """
    Predict future bounding boxes using constant velocity.

    x_future = x + dx * steps
    y_future = y + dy * steps

    Bounding-box dimensions remain unchanged.
    """

    predicted = boxes.copy()

    displacement = velocities * steps

    predicted[:, [0, 2]] += displacement[:, 0, None]
    predicted[:, [1, 3]] += displacement[:, 1, None]

    return predicted


# ============================================================
# CHECK WHETHER PLATE SET REMAINS STABLE
# ============================================================

def checkSet(prev_boxes, curr_boxes, max_distance=80):
    """
    Check whether all plates can be matched one-to-one.

    Returns:
        matched_indices if stable
        None otherwise
    """

    return match_plates(
        prev_boxes,
        curr_boxes,
        max_distance
    )


# ============================================================
# HSV COLOR DETECTION
# ============================================================

def color_detection(frame):
    """
    Runs on CPU.

    Receives the original frame.
    Does not modify the original frame.

    Returns:
        List of (color, bounding_box) tuples.
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

        mask = np.zeros(
            hsv.shape[:2],
            dtype=np.uint8
        )

        for lower, upper in ranges:
            mask |= cv2.inRange(
                hsv,
                lower,
                upper
            )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            if cv2.contourArea(contour) > 100:

                x, y, w, h = cv2.boundingRect(contour)

                detections.append(
                    (color, (x, y, w, h))
                )

    return detections


# ============================================================
# MATCH PLATES TO COLOR REGIONS
# ============================================================

def grouping(image, plate_boxes, enemyColor,
             allyColor, maxPixDistance):
    """
    Match YOLO or predicted bounding boxes to nearby color
    regions.

    Returns:
        List of (team, center) tuples.

    Example:
        [
            ("enemy", (320, 240)),
            ("ally", (510, 180))
        ]
    """

    enemyColor = enemyColor.lower()
    allyColor = allyColor.lower()

    available_colors = {"red", "blue"}

    if enemyColor not in available_colors:
        raise ValueError(
            "enemyColor must be 'red' or 'blue'"
        )

    if allyColor not in available_colors:
        raise ValueError(
            "allyColor must be 'red' or 'blue'"
        )

    if enemyColor == allyColor:
        raise ValueError(
            "enemyColor and allyColor must be different"
        )

    if maxPixDistance < 0:
        raise ValueError(
            "maxPixDistance must be non-negative"
        )

    if image is None or plate_boxes is None:
        return []

    if len(plate_boxes) == 0:
        return []

    # Detect color regions.
    color_boxes = color_detection(image)

    if not color_boxes:
        return []

    # Keep relevant colors.
    color_boxes = [
        (color, box)
        for color, box in color_boxes
        if color in (enemyColor, allyColor)
    ]

    if not color_boxes:
        return []

    candidates = []

    # Compare every plate against every color region.
    for plate_index, plate_box in enumerate(plate_boxes):

        px1, py1, px2, py2 = plate_box

        center = (
            int((px1 + px2) / 2),
            int((py1 + py2) / 2)
        )

        for region_index, (
            color, (rx, ry, rw, rh)
        ) in enumerate(color_boxes):

            # Distance between rectangles.
            dx = max(
                px1 - (rx + rw),
                rx - px2,
                0
            )

            dy = max(
                py1 - (ry + rh),
                ry - py2,
                0
            )

            distance = (
                dx * dx + dy * dy
            ) ** 0.5

            if distance <= maxPixDistance:

                team = (
                    "enemy"
                    if color == enemyColor
                    else "ally"
                )

                candidates.append(
                    (
                        distance,
                        plate_index,
                        region_index,
                        team,
                        center
                    )
                )

    # Greedy nearest-neighbor matching.
    matches = []

    matched_plates = set()
    matched_regions = set()

    for (
        _,
        plate_index,
        region_index,
        team,
        center
    ) in sorted(candidates):

        if plate_index in matched_plates:
            continue

        if region_index in matched_regions:
            continue

        matched_plates.add(plate_index)
        matched_regions.add(region_index)

        matches.append(
            (team, center)
        )

    return matches


# ============================================================
# VIDEO TRACKING WITH CONDITIONAL INFERENCE SKIPPING
# ============================================================

def track_video(model, video_path, enemyColor,
                allyColor, maxPixDistance):

    video_path = Path(video_path)

    if not video_path.is_absolute():
        video_path = root / video_path

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    # Asynchronous color grouping.
    executor = ThreadPoolExecutor(max_workers=1)

    pending = deque()

    frame_count = 0
    inference_count = 0
    predicted_count = 0

    start_time = time.perf_counter()

    # Last actual YOLO detection.
    last_boxes = None
    last_detection_frame = None

    # Per-frame velocities aligned with last_boxes.
    velocities = None

    # If positive, skip this many upcoming frames.
    skip_remaining = 0

    try:
        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # =================================================
            # CASE 1: SKIP YOLO AND PREDICT
            # =================================================

            if (
                skip_remaining > 0
                and velocities is not None
                and last_boxes is not None
            ):

                steps = (
                    frame_count - last_detection_frame
                )

                plate_boxes = predict_boxes(
                    last_boxes,
                    velocities,
                    steps=steps
                )

                # Clamp predicted boxes to frame boundaries.
                frame_height, frame_width = frame.shape[:2]

                plate_boxes[:, [0, 2]] = np.clip(
                    plate_boxes[:, [0, 2]],
                    0,
                    frame_width
                )

                plate_boxes[:, [1, 3]] = np.clip(
                    plate_boxes[:, [1, 3]],
                    0,
                    frame_height
                )

                skip_remaining -= 1
                predicted_count += 1

                mode = "PREDICT"

            # =================================================
            # CASE 2: RUN YOLO
            # =================================================

            else:

                result = model.predict(
                    source=frame,
                    imgsz=IMGSZ,
                    conf=CONF,
                    save=False,
                    device=DEVICE,
                    verbose=False
                )[0]

                inference_count += 1

                plate_boxes = (
                    result.boxes.xyxy
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

                mode = "YOLO"

                # ---------------------------------------------
                # CHECK WHETHER PLATES ARE STABLE
                # ---------------------------------------------

                if last_boxes is not None:

                    dt = (
                        frame_count - last_detection_frame
                    )

                    # If we already have velocities, use predicted
                    # locations to assist matching after skipped frames.
                    if velocities is not None:

                        reference_boxes = predict_boxes(
                            last_boxes,
                            velocities,
                            steps=dt
                        )

                    else:

                        reference_boxes = last_boxes

                    matched_indices = checkSet(
                        reference_boxes,
                        plate_boxes,
                        MAX_TRACK_DISTANCE
                    )

                    if matched_indices is not None:

                        # Calculate displacement against actual
                        # previous YOLO detections, using the
                        # correct plate correspondence.
                        velocities = calcChange(
                            last_boxes,
                            plate_boxes,
                            matched_indices,
                            dt=dt
                        )

                        # Stable set: skip the next frame.
                        skip_remaining = SKIP_FRAMES

                    else:

                        # Plate set changed or association failed.
                        # Run YOLO again on the next frame.
                        velocities = None
                        skip_remaining = 0

                else:

                    # First detection: no previous frame available.
                    velocities = None
                    skip_remaining = 0

                # Save current actual YOLO detections.
                last_boxes = plate_boxes.copy()
                last_detection_frame = frame_count

            # =================================================
            # ASYNCHRONOUS COLOR GROUPING
            # =================================================

            future = executor.submit(
                grouping,
                frame,
                plate_boxes.copy(),
                enemyColor,
                allyColor,
                maxPixDistance
            )

            pending.append(
                (frame_count, mode, future)
            )

            # Limit outstanding jobs.
            if len(pending) > MAX_PENDING:

                frame_id, frame_mode, completed = (
                    pending.popleft()
                )

                matches = completed.result()

                for team, center in matches:

                    print(
                        f"Frame: {frame_id} | "
                        f"Mode: {frame_mode} | "
                        f"Team: {team} | "
                        f"Center: {center}"
                    )

        # =====================================================
        # COLLECT REMAINING ASYNC JOBS
        # =====================================================

        while pending:

            frame_id, frame_mode, completed = (
                pending.popleft()
            )

            matches = completed.result()

            for team, center in matches:

                print(
                    f"Frame: {frame_id} | "
                    f"Mode: {frame_mode} | "
                    f"Team: {team} | "
                    f"Center: {center}"
                )

    finally:

        executor.shutdown(wait=True)
        cap.release()

    # =========================================================
    # PERFORMANCE STATISTICS
    # =========================================================

    elapsed = time.perf_counter() - start_time

    if elapsed > 0 and frame_count > 0:

        fps = frame_count / elapsed

        skip_percent = (
            predicted_count / frame_count
        ) * 100

        print("\n" + "=" * 55)

        print(f"Video: {video_path.name}")

        print(f"Total frames: {frame_count}")

        print(f"YOLO inference calls: {inference_count}")

        print(f"Predicted frames: {predicted_count}")

        print(f"Inference skipped: {skip_percent:.2f}%")

        print(f"Total processing time: {elapsed:.2f}s")

        print(f"Average processing FPS: {fps:.2f}")

        print("=" * 55 + "\n")

    return frame_count


# ============================================================
# MAIN
# ============================================================

def main():

    model_path = (
        root /
        "runs/detect/[LARGE]Robomasters.v6i.yolo26_EPOCH100/"
        "weights/best.engine"
    )

    model = YOLO(str(model_path))

    enemyColor = "red"
    allyColor = "blue"

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