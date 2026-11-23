from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
import time

import cv2
import numpy as np
from ultralytics import YOLO


root = Path(__file__).resolve().parent


def track_camera(model, camera_index, enemyColor, allyColor, maxPixDistance):
    frame_count = 0
    start_time = time.perf_counter()
    executor = ThreadPoolExecutor(max_workers=1)
    pending = deque()

    try:
        results = model.predict(
            source=camera_index,
            imgsz=418 + 32 * 5,
            conf=0.1,
            tracker=str(root / "byteTrackConfig.yaml"),
            save=False,
            device=0,
            stream=True,
        )
        with closing(results):
            for result in results:
                frame_count += 1
                plate_boxes = result.boxes.xyxy.cpu().numpy()

                future = executor.submit(
                    grouping,
                    result.orig_img,
                    plate_boxes,
                    enemyColor,
                    allyColor,
                    maxPixDistance,
                )
                pending.append((frame_count, future))

                if len(pending) > 2:
                    frame_id, completed = pending.popleft()
                    for team, center in completed.result():
                        print(
                            f"Frame: {frame_id} | "
                            f"Team: {team} | "
                            f"Center: {center}"
                        )
    except KeyboardInterrupt:
        print("\nStopping camera tracking.")
    finally:
        executor.shutdown(wait=True)

    while pending:
        frame_id, completed = pending.popleft()
        for team, center in completed.result():
            print(
                f"Frame: {frame_id} | "
                f"Team: {team} | "
                f"Center: {center}"
            )

    elapsed = time.perf_counter() - start_time
    if elapsed > 0:
        print(
            f"Tracking speed for camera {camera_index}: "
            f"{frame_count / elapsed:.2f} FPS "
            f"({frame_count} frames over {elapsed:.2f}s)"
        )

    return frame_count


def color_detection(frame):
    """Detect red and blue regions in a BGR frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_ranges = {
        "red": [
            (np.array([0, 100, 100]), np.array([10, 255, 255])),
            (np.array([170, 100, 100]), np.array([180, 255, 255])),
        ],
        "blue": [
            (np.array([100, 100, 100]), np.array([130, 255, 255])),
        ],
    }

    detections = []
    for color, ranges in color_ranges.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(hsv, lower, upper)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                x, y, width, height = cv2.boundingRect(contour)
                detections.append((color, (x, y, width, height)))

    return detections


def grouping(image, plate_boxes, enemyColor, allyColor, maxPixDistance):
    """Match YOLO boxes to nearby team-colored regions."""
    enemyColor = enemyColor.lower()
    allyColor = allyColor.lower()
    available_colors = {"red", "blue"}
    if enemyColor not in available_colors or allyColor not in available_colors:
        raise ValueError("enemyColor and allyColor must be 'red' or 'blue'")
    if enemyColor == allyColor:
        raise ValueError("enemyColor and allyColor must be different")
    if maxPixDistance < 0:
        raise ValueError("maxPixDistance must be non-negative")
    if image is None or plate_boxes is None or len(plate_boxes) == 0:
        return []

    color_boxes = [
        (color, box)
        for color, box in color_detection(image)
        if color in (enemyColor, allyColor)
    ]
    if not color_boxes:
        return []

    candidates = []
    for plate_index, plate_box in enumerate(plate_boxes):
        px1, py1, px2, py2 = plate_box
        center = (int((px1 + px2) / 2), int((py1 + py2) / 2))

        for region_index, (color, (rx, ry, rw, rh)) in enumerate(color_boxes):
            dx = max(px1 - (rx + rw), rx - px2, 0)
            dy = max(py1 - (ry + rh), ry - py2, 0)
            distance = (dx * dx + dy * dy) ** 0.5

            if distance <= maxPixDistance:
                team = "enemy" if color == enemyColor else "ally"
                candidates.append(
                    (distance, plate_index, region_index, team, center)
                )

    matches = []
    matched_plates = set()
    matched_regions = set()
    for _, plate_index, region_index, team, center in sorted(candidates):
        if plate_index in matched_plates or region_index in matched_regions:
            continue

        matched_plates.add(plate_index)
        matched_regions.add(region_index)
        matches.append((team, center))

    return matches


def main():
    model_path = (
        root / "runs/detect/[LARGE]Robomasters.v6i.yolo26_EPOCH100/weights/best.engine"
    )
    model = YOLO(str(model_path))

    track_camera(
        model,
        camera_index=0,
        enemyColor="red",
        allyColor="blue",
        maxPixDistance=80,
    )


if __name__ == "__main__":
    main()
