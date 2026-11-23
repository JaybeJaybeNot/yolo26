
from pathlib import Path
import time
from ultralytics import YOLO
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from collections import deque


# IMPORTANT:
# Color detection and grouping run asynchronously.
# Results may be retrieved a few frames after YOLO processes them,
# but each result is drawn onto its corresponding original frame.

skip = 1
root = Path(__file__).resolve().parent

videos = [
    "[BLUR_TEST]RoboMasterTestVideo3.mp4",
    "RoboMasterTestVideo2.mp4",
]


class plate:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def move(self, dx, dy):
        self.x += dx
        self.y += dy


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


def draw_matches(frame, matches):
    """
    Draw enemy and ally bounding boxes onto a frame.

    Parameters:
        frame: Original BGR frame.
        matches: List of (team, box) tuples.

    Returns:
        Annotated frame.
    """

    for team, box in matches:

        x1, y1, x2, y2 = map(int, box)

        if team == "enemy":
            color = (0, 0, 255)  # Red in BGR
            label = "ENEMY"

        else:
            color = (255, 0, 0)  # Blue in BGR
            label = "ALLY"

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # Calculate label dimensions
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness
        )

        # Place label above the bounding box
        label_y = max(y1, text_height + 10)

        # Draw filled background behind label
        cv2.rectangle(
            frame,
            (x1, label_y - text_height - 8),
            (x1 + text_width + 8, label_y + baseline - 4),
            color,
            -1
        )

        # Draw label text
        cv2.putText(
            frame,
            label,
            (x1 + 4, label_y - 4),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    return frame


def track_video(model, video_path, enemyColor, allyColor, maxPixDistance, save=True):

    video_path = Path(video_path)

    if not video_path.is_absolute():
        video_path = root / video_path

    frame_count = 0
    start_time = time.perf_counter()

    # Output video configuration
    writer = None

    if save:

        output_dir = root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{video_path.stem}_grouped.mp4"

        # Read video metadata only; no frames are decoded here.
        cap = cv2.VideoCapture(str(video_path))

        fps = cap.get(cv2.CAP_PROP_FPS)

        cap.release()

        if fps <= 0:
            fps = 30.0

        print(f"Saving annotated video to: {output_path}")
        print(f"Output FPS: {fps}")

    # One CPU worker for color detection and grouping
    executor = ThreadPoolExecutor(max_workers=1)

    # Limit outstanding jobs to prevent memory accumulation
    pending = deque()

    results = model.predict(
        source=video_path,
        imgsz=640,
        conf=0.1,
        tracker=str(root / "byteTrackConfig.yaml"),
        save=False,
        device=0,
        stream=True,
        vid_stride = skip
    )

    try:

        for result in results:

            frame_count += 1
            # Extract YOLO bounding boxes
            plate_boxes = result.boxes.xyxy.cpu().numpy()
            # Access the SAME frame YOLO already decoded
            frame = result.orig_img

            # Create the output writer on the first frame
            if save and writer is None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    fps,
                    (width, height)
                )
                if not writer.isOpened():
                    raise RuntimeError(
                        f"Could not create output video: {output_path}"
                    )

            # Submit grouping asynchronously
            future = executor.submit(
                grouping,
                frame,
                plate_boxes,
                enemyColor,
                allyColor,
                maxPixDistance
            )

            # Keep the original frame alongside its future
            pending.append((frame_count, frame.copy(), future))

            # Limit pending jobs to 2
            if len(pending) > 2:

                frame_id, completed_frame, completed = pending.popleft()

                matches = completed.result()

                # Draw custom labels onto the corresponding frame
                if save:
                    completed_frame = draw_matches(
                        completed_frame,
                        matches
                    )

                    writer.write(completed_frame)

                # Print grouping results
                for team, box in matches:

                    print(
                        f"Frame: {frame_id} | "
                        f"Team: {team} | "
                        f"Box: {box}"
                    )

        # Collect remaining jobs
        while pending:

            frame_id, completed_frame, completed = pending.popleft()

            matches = completed.result()

            if save:
                completed_frame = draw_matches(
                    completed_frame,
                    matches
                )

                writer.write(completed_frame)

            for team, box in matches:

                print(
                    f"Frame: {frame_id} | "
                    f"Team: {team} | "
                    f"Box: {box}"
                )

    finally:

        executor.shutdown(wait=True)

        if writer is not None:
            writer.release()

    elapsed = time.perf_counter() - start_time

    if elapsed > 0:
        print(f"Tracking speed for {video_path.name}: "
            f"{frame_count / elapsed:.2f} FPS "
            f"({frame_count} frames over {elapsed:.2f}s)"
        )

    if save:
        print(f"Saved annotated video: {output_path}")

    return frame_count


def color_detection(frame):
    """
    Detect red and blue regions in a BGR frame.

    Returns:
        List of (color, (x, y, w, h)) tuples.
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

                detections.append(
                    (color, (x, y, w, h))
                )

    return detections


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
        List of (team, box) tuples.

        Example:
            [
                ("enemy", (100, 120, 160, 180)),
                ("ally", (300, 200, 360, 260))
            ]
    """

    # Validate inputs
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

    # Detect color regions
    color_boxes = color_detection(image)

    if not color_boxes:
        return []

    # Keep only relevant team colors
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

        for region_index, (color, (rx, ry, rw, rh)) in enumerate(color_boxes):

            # Calculate distance between rectangles
            dx = max(px1 - (rx + rw), rx - px2, 0)
            dy = max(py1 - (ry + rh), ry - py2, 0)

            distance = (dx * dx + dy * dy) ** 0.5

            if distance <= maxPixDistance:

                team = "enemy" if color == enemyColor else "ally"

                candidates.append(
                    (
                        distance,
                        plate_index,
                        region_index,
                        team,
                        tuple(map(int, plate_box))
                    )
                )

    # Greedy nearest-neighbor matching
    matches = []

    matched_plates = set()
    matched_regions = set()

    for _, plate_index, region_index, team, box in sorted(candidates):

        if plate_index in matched_plates or region_index in matched_regions:
            continue

        matched_plates.add(plate_index)
        matched_regions.add(region_index)

        matches.append(
            (team, box)
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

    # Enable or disable annotated video output
    save = True

    for video in videos:

        track_video(
            model,
            video,
            enemyColor,
            allyColor,
            maxPixDistance,
            save
        )


if __name__ == "__main__":
    main()
