import argparse
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent

videos = [
    "RoboMasterTestVideo1.mp4",
    "RoboMasterTestVideo2.mp4",
]
# Replace these names with videos that show a stationary or moving chessboard.
# Gameplay, tracking, and test footage do not contain a calibration pattern.


def find_chessboard(frame, pattern_size):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(
            gray,
            pattern_size,
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
    else:
        found, corners = cv2.findChessboardCorners(
            gray,
            pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_FAST_CHECK,
        )

    if found:
        corners = cv2.cornerSubPix(
            gray,
            corners,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
    return found, corners


def calibrate_video_files(video_paths, pattern_size, square_size, frame_stride, max_frames):
    object_points_template = np.zeros(
        (pattern_size[0] * pattern_size[1], 3),
        dtype=np.float64,
    )
    object_points_template[:, :2] = np.mgrid[
        0:pattern_size[0],
        0:pattern_size[1],
    ].T.reshape(-1, 2)
    object_points_template *= square_size

    object_point_sets = []
    image_point_sets = []
    image_size = None
    sampled_frames = 0

    for video_path in video_paths:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            print(f"Skipping unreadable video: {video_path}")
            continue

        frame_index = 0
        video_frames = 0
        print(f"Scanning {video_path}")
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % frame_stride != 0:
                frame_index += 1
                continue

            if image_size is None:
                image_size = (frame.shape[1], frame.shape[0])
            elif image_size != (frame.shape[1], frame.shape[0]):
                print(
                    f"Skipping frame with different resolution "
                    f"{frame.shape[1]}x{frame.shape[0]}: {video_path}"
                )
                frame_index += 1
                continue

            found, corners = find_chessboard(frame, pattern_size)
            if found:
                object_point_sets.append(object_points_template.copy())
                image_point_sets.append(corners)
                video_frames += 1
                sampled_frames += 1
                print(
                    f"  Added chessboard from frame {frame_index} "
                    f"({sampled_frames} total)"
                )
            if sampled_frames >= max_frames:
                break
            frame_index += 1
        capture.release()
        print(f"  Found {video_frames} usable frames")
        if sampled_frames >= max_frames:
            break

    if len(object_point_sets) < 10:
        raise RuntimeError(
            "Not enough chessboard views were found "
            f"(found {len(object_point_sets)}, need at least 10). "
            "The current videos do not appear to contain the configured "
            f"{pattern_size[0]}x{pattern_size[1]} inner-corner chessboard. "
            "Replace the entries in the videos list with calibration videos "
            "that show a checkerboard from several angles and distances."
        )

    rms, camera_matrix, dist_coeffs, rotations, translations = cv2.calibrateCamera(
        object_point_sets,
        image_point_sets,
        image_size,
        None,
        None,
    )
    return (
        rms,
        camera_matrix,
        dist_coeffs,
        rotations,
        translations,
        len(object_point_sets),
        image_size,
    )


def save_calibration(output_path, rms, camera_matrix, dist_coeffs, image_size):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path.with_suffix(".npz"),
        K=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_width=image_size[0],
        image_height=image_size[1],
        rms=rms,
    )

    yaml_path = output_path.with_suffix(".yaml")
    fs = cv2.FileStorage(str(yaml_path), cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", camera_matrix)
    fs.write("distortion_coefficients", dist_coeffs)
    fs.write("image_width", image_size[0])
    fs.write("image_height", image_size[1])
    fs.write("reprojection_error", rms)
    fs.release()
    return output_path.with_suffix(".npz"), yaml_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate a camera using chessboards observed in one or more videos."
    )
    parser.add_argument(
        "--pattern",
        default="9x6",
        help="Chessboard inner-corner pattern as columns x rows (default: 9x6).",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=0.025,
        help="Physical checkerboard square size in meters (default: 0.025).",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=30,
        help="Process every Nth frame (default: 30).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Maximum number of chessboard views to use (default: 300).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="camera_calibration",
        help="Output basename (default: camera_calibration).",
    )
    return parser, parser.parse_args()


def main():
    parser, args = parse_args()

    if "x" not in args.pattern.lower():
        parser.error("--pattern must be formatted as columns x rows, for example 9x6")
    try:
        columns, rows = (int(value) for value in args.pattern.lower().split("x"))
    except ValueError:
        parser.error("--pattern must be formatted as columns x rows, for example 9x6")
    if columns < 2 or rows < 2:
        parser.error("Chessboard pattern dimensions must be at least 2x2")
    if args.square_size <= 0:
        parser.error("--square-size must be positive")
    if args.frame_stride <= 0:
        parser.error("--frame-stride must be positive")
    if args.max_frames < 10:
        parser.error("--max-frames must be at least 10")

    video_paths = [ROOT / video_name for video_name in videos]
    video_paths = [path for path in video_paths if path.is_file()]
    if not video_paths:
        parser.error(
            "None of the defined video files exist: " + ", ".join(videos)
        )

    rms, camera_matrix, dist_coeffs, _, _, view_count, image_size = calibrate_video_files(
        video_paths,
        (columns, rows),
        args.square_size,
        args.frame_stride,
        args.max_frames,
    )
    npz_path, yaml_path = save_calibration(
        args.output,
        rms,
        camera_matrix,
        dist_coeffs,
        image_size,
    )

    print(f"\nUsed {view_count} chessboard views at {image_size[0]}x{image_size[1]}")
    print(f"Reprojection RMS error: {rms:.6f} pixels")
    print("Camera matrix K:")
    print(camera_matrix)
    print("Distortion coefficients:")
    print(dist_coeffs.ravel())
    print(f"Saved NumPy calibration: {npz_path}")
    print(f"Saved OpenCV YAML: {yaml_path}")


if __name__ == "__main__":
    main()
