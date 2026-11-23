from pathlib import Path

import cv2
from ultralytics import YOLO


root = Path(__file__).resolve().parent
videos = [
    "[BLUR_TEST]RoboMasterTestVideo3.mp4",
    #"RoboMasterTestVideo1.mp4",
    "RoboMasterTestVideo2.mp4",
]


def clean_track(model_path, video_path):
    model = YOLO(str(model_path))
    video_path = Path(video_path)
    if not video_path.is_absolute():
        video_path = root / video_path

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Could not read video properties: {video_path}")

    output_dir = root / "runs" / "predict"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_path.stem}_classified.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")

    try:
        results = model.track(
            source=str(video_path),
            imgsz=436,
            conf=0.01,
            tracker=str(root / "byteTrackConfig.yaml"),
            stream=True,
            persist=True,
        )

        for result in results:
            frame = result.orig_img.copy()
            matches = grouping(
                result, enemy_color="red", ally_color="blue", max_pixel_distance=20
            )
            for team, box, track_id in matches:
                x1, y1, x2, y2 = box
                color = (0, 0, 255) if team == "enemy" else (255, 0, 0)
                label = team.upper()
                if track_id is not None:
                    label += f" ID {track_id}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

            writer.write(frame)
    finally:
        writer.release()

    print(f"Classified tracking video saved to: {output_path}")


def detect_color_regions(image):
    """Find sufficiently large red and blue regions in a full frame."""
    if image is None or image.size == 0:
        return []

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    masks = {
        "red": cv2.bitwise_or(
            cv2.inRange(hsv, (0, 80, 50), (10, 255, 255)),
            cv2.inRange(hsv, (170, 80, 50), (180, 255, 255)),
        ),
        "blue": cv2.inRange(hsv, (90, 80, 50), (130, 255, 255)),
    }

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    minimum_area = max(100, int(image.shape[0] * image.shape[1] * 0.0001))
    regions = []
    for color, mask in masks.items():
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        regions.extend(
            (color, cv2.boundingRect(contour))
            for contour in contours
            if cv2.contourArea(contour) >= minimum_area
        )
    return regions


def grouping(frame, enemy_color, ally_color, max_pixel_distance):
    """Associate nearby red/blue regions with tracked plate boxes."""
    enemy_color = enemy_color.lower()
    ally_color = ally_color.lower()
    if enemy_color not in {"red", "blue"} or ally_color not in {"red", "blue"}:
        raise ValueError("enemy_color and ally_color must be 'red' or 'blue'")
    if enemy_color == ally_color:
        raise ValueError("enemy_color and ally_color must be different")
    if max_pixel_distance < 0:
        raise ValueError("max_pixel_distance must be non-negative")

    boxes = frame.boxes
    if frame.orig_img is None or boxes is None:
        return []

    plate_boxes = boxes.xyxy.cpu().numpy()
    if len(plate_boxes) == 0:
        return []

    regions = [
        (color, box)
        for color, box in detect_color_regions(frame.orig_img)
        if color in (enemy_color, ally_color)
    ]
    if not regions:
        return []

    ids = boxes.id
    track_ids = (
        ids.cpu().numpy().astype(int).tolist()
        if ids is not None
        else [None] * len(plate_boxes)
    )

    candidates = []
    for plate_index, plate in enumerate(plate_boxes):
        x1, y1, x2, y2 = plate
        for region_index, (region_color, (rx, ry, rw, rh)) in enumerate(regions):
            dx = max(x1 - (rx + rw), rx - x2, 0)
            dy = max(y1 - (ry + rh), ry - y2, 0)
            distance = (dx * dx + dy * dy) ** 0.5
            if distance <= max_pixel_distance:
                team = "enemy" if region_color == enemy_color else "ally"
                box = tuple(map(int, (x1, y1, x2, y2)))
                candidates.append((distance, plate_index, region_index, team, box))

    matches = []
    used_plates = set()
    used_regions = set()
    for _, plate_index, region_index, team, box in sorted(candidates):
        if plate_index in used_plates or region_index in used_regions:
            continue
        used_plates.add(plate_index)
        used_regions.add(region_index)
        matches.append((team, box, track_ids[plate_index]))
    return matches


def main():
    model_path = (
        root
        / "runs/detect/[LARGE]Robomasters.v6i.yolo26_EPOCH100/weights/best.pt"
    )
    for video in videos:
        clean_track(model_path, root / video)


if __name__ == "__main__":
    main()
