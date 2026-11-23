from pathlib import Path
import time

import cv2
import numpy as np


root = Path(__file__).resolve().parent
videos = [
    "[BLUR_TEST]RoboMasterTestVideo3.mp4",
    # "RoboMasterTestVideo1.mp4",
    "RoboMasterTestVideo2.mp4",
]


def watershed_frame(frame, distance_ratio=0.4):
    """Segment foreground objects and overlay watershed boundaries."""
    segmented, _ = watershed_outputs(frame, distance_ratio)
    return segmented


def watershed_outputs(frame, distance_ratio):
    """Return the boundary overlay and a black frame containing only boundaries."""
    if frame is None or frame.size == 0:
        raise ValueError("frame must be a non-empty image")
    if not 0 < distance_ratio < 1:
        raise ValueError("distance_ratio must be between 0 and 1")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_foreground = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_background = cv2.dilate(sure_foreground, kernel, iterations=3)

    distance = cv2.distanceTransform(sure_foreground, cv2.DIST_L2, 5)
    max_distance = float(distance.max())
    if max_distance == 0:
        return frame.copy(), np.zeros_like(frame)

    _, sure_foreground = cv2.threshold(
        distance, distance_ratio * max_distance, 255, cv2.THRESH_BINARY
    )
    sure_foreground = sure_foreground.astype(np.uint8)
    unknown = cv2.subtract(sure_background, sure_foreground)

    _, markers = cv2.connectedComponents(sure_foreground)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(frame.copy(), markers)

    boundary = markers == -1
    boundary[[0, -1], :] = False
    boundary[:, [0, -1]] = False
    segmented = frame.copy()
    segmented[boundary] = (0, 0, 255)
    outline = np.zeros_like(frame)
    outline[boundary] = (255, 255, 255)
    return segmented, outline


def process_video(video_path, output_dir=None, distance_ratio=0.4):
    """Apply distance-transform-seeded watershed segmentation to a video."""
    video_path = Path(video_path)
    if not video_path.is_absolute():
        video_path = root / video_path
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_dir = Path(output_dir) if output_dir is not None else root / "runs" / "predict"
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_path.stem}_watershed.mp4"
    outline_path = output_dir / f"{video_path.stem}_watershed_outline.mp4"

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Could not read video properties: {video_path}")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    outline_writer = cv2.VideoWriter(
        str(outline_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not outline_writer.isOpened():
        capture.release()
        writer.release()
        raise RuntimeError(f"Could not create outline video: {outline_path}")

    frame_count = 0
    frame_intervals = []
    previous_frame_time = None
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            segmented, outline = watershed_outputs(frame, distance_ratio)
            writer.write(segmented)
            outline_writer.write(outline)
            frame_count += 1
            current_frame_time = time.perf_counter()
            if previous_frame_time is not None:
                frame_intervals.append(current_frame_time - previous_frame_time)
            previous_frame_time = current_frame_time
    finally:
        capture.release()
        writer.release()
        outline_writer.release()

    if frame_count == 0:
        raise RuntimeError(f"No frames could be read from video: {video_path}")

    print(f"Processed {frame_count} frames: {output_path}")
    print(f"Outline-only video saved to: {outline_path}")
    if frame_intervals:
        average_interval = sum(frame_intervals) / len(frame_intervals)
        measured_fps = 1 / average_interval
        print(f"Average time between processed frames: {average_interval:.4f} seconds")
        print(f"Measured processing rate: {measured_fps:.2f} frames per second")
    else:
        print("Measured processing rate: unavailable (fewer than two frames processed)")
    return output_path


def main():
    for video in videos:
        process_video(video)


if __name__ == "__main__":
    main()
