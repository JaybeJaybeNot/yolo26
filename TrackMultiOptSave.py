import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from ultralytics import YOLO

from TrackMultiOpt import grouping, root, skip, videos


def track_video(model, video_path, enemy_color, ally_color, max_pixel_distance):
    video_path = Path(video_path)
    if not video_path.is_absolute():
        video_path = root / video_path

    frame_plates = {}
    frame_count = 0
    inference_ms = 0.0
    start_time = time.perf_counter()
    executor = ThreadPoolExecutor(max_workers=1)
    pending = deque()

    results = model.predict(
        source=video_path,
        imgsz=418 + 32 * 5,
        conf=0.1,
        tracker=str(root / "byteTrackConfig.yaml"),
        save=False,
        device=0,
        stream=True,
        vid_stride=skip,
        #show = True
    )

    try:
        for result in results:
            frame_count += 1
            source_frame_index = (frame_count - 1) * skip

            plate_boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id
            track_ids = (
                track_ids.cpu().numpy().astype(int).tolist()
                if track_ids is not None
                else [None] * len(plate_boxes)
            )
            frame_plates[source_frame_index] = [
                {
                    "id": track_id,
                    "box": [float(value) for value in box],
                }
                for track_id, box in zip(track_ids, plate_boxes)
            ]

            inference_ms += result.speed.get("inference", 0.0)

            future = executor.submit(
                grouping,
                result.orig_img.copy(),
                plate_boxes,
                enemy_color,
                ally_color,
                max_pixel_distance,
            )
            pending.append((frame_count, future))

            if len(pending) > 2:
                frame_id, completed = pending.popleft()
                for team, center in completed.result():
                    print(f"Frame: {frame_id} | Team: {team} | Center: {center}")

        while pending:
            frame_id, completed = pending.popleft()
            for team, center in completed.result():
                print(f"Frame: {frame_id} | Team: {team} | Center: {center}")
    finally:
        executor.shutdown(wait=True)

    elapsed = time.perf_counter() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0.0
    inference_fps = frame_count / (inference_ms / 1000) if inference_ms > 0 else 0.0

    print(
        f"Tracking speed for {video_path.name}: {fps:.2f} FPS "
        f"({frame_count} inference frames)"
    )
    print(
        f"Inference time for {video_path.name}: {inference_ms:.2f} ms total "
        f"({inference_fps:.2f} inference FPS)"
    )

    return {
        "video_path": video_path,
        "frame_plates": frame_plates,
        "inference_ms": inference_ms,
        "fps": fps,
        "inference_fps": inference_fps,
        "inference_frames": frame_count,
    }


def save_plates_video(tracking_result):
    video_path = tracking_result["video_path"]
    output_dir = root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_path.stem}_plates.mp4"
    plates_path = output_dir / f"{video_path.stem}_plates.json"

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 30.0

    writer = None
    frame_index = 0
    try:
        with plates_path.open("w", encoding="utf-8") as data_file:
            data_file.write(
                json.dumps(
                    {
                        "video": video_path.name,
                        "fps": source_fps,
                        "inference_ms": tracking_result["inference_ms"],
                        "inference_fps": tracking_result["inference_fps"],
                    }
                )[:-1]
                + ',"frames":['
            )

            while True:
                success, frame = capture.read()
                if not success:
                    break

                sample_index = (frame_index // skip) * skip
                plates = tracking_result["frame_plates"].get(sample_index, [])
                for plate_data in plates:
                    x1, y1, x2, y2 = map(int, plate_data["box"])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    cv2.circle(frame, center, 4, (0, 255, 0), -1)
                    label = (
                        f"ID: {plate_data['id']}"
                        if plate_data["id"] is not None
                        else "PLATE"
                    )
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        source_fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(
                            f"Could not create output video: {output_path}"
                        )

                writer.write(frame)
                if frame_index:
                    data_file.write(",")
                data_file.write(
                    json.dumps({"frame": frame_index, "plates": plates})
                )
                frame_index += 1

            data_file.write("]}\n")
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    print(f"Saved annotated video: {output_path}")
    print(f"Saved per-frame plate locations: {plates_path}")
    return output_path, plates_path


def main():
    model_path = (
        root / "runs/detect/[LARGE]Robomasters.v6i.yolo26_EPOCH100/weights/best.engine"
    )
    model = YOLO(str(model_path))

    for video in videos:
        tracking_result = track_video(model, video, "red", "blue", 80)
        save_plates_video(tracking_result)


if __name__ == "__main__":
    main()