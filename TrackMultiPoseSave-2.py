import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import TrackMultiPose as pose_tracking
from TrackMultiPose import calculateTrajectory,get_camera_matrix, get_object_points, root, videos, imgsz

def drawTrajectory():
    
PLATE_SIDE_COLORS = (
    (0, 0, 255),
    (0, 165, 255),
    (0, 255, 255),
    (255, 0, 0),
)
MAP_MAX_DEPTH_METERS = 10.0
MAP_HALF_WIDTH_METERS = 2.5


def draw_robot_radar(frame, robots):
    frame_height, frame_width = frame.shape[:2]
    map_width = max(180, min(260, round(frame_width * 0.23)))
    map_height = max(150, min(220, round(frame_height * 0.30)))
    margin = 16
    x1 = frame_width - margin - map_width
    y1 = margin
    x2 = frame_width - margin
    y2 = y1 + map_height

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (180, 180, 180), 1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    cv2.putText(
        frame,
        "Robot radar",
        (x1 + 10, y1 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    map_left = x1 + 26
    map_right = x2 - 26
    map_bottom = y2 - 28
    map_top = y1 + 42
    camera_tip = ((map_left + map_right) // 2, map_bottom)
    map_half_width = (map_right - map_left) // 2
    depth_height = map_bottom - map_top

    cone_outline = np.array(
        [
            camera_tip,
            (camera_tip[0] - map_half_width, map_top),
            (camera_tip[0] + map_half_width, map_top),
        ],
        dtype=np.int32,
    )
    cv2.polylines(
        frame,
        [cone_outline],
        isClosed=True,
        color=(100, 100, 100),
        thickness=1,
        lineType=cv2.LINE_AA,
    )

    max_depth = MAP_MAX_DEPTH_METERS
    grid_step = 1.0
    grid_depth = grid_step
    while grid_depth <= max_depth:
        normalized_depth = grid_depth / max_depth
        grid_y = round(map_bottom - normalized_depth * depth_height)
        grid_half_width = round(map_half_width * normalized_depth)
        cv2.line(
            frame,
            (camera_tip[0] - grid_half_width, grid_y),
            (camera_tip[0] + grid_half_width, grid_y),
            (75, 75, 75),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"{grid_depth:g}m",
            (camera_tip[0] + 7, grid_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (170, 170, 170),
            1,
            cv2.LINE_AA,
        )
        grid_depth += grid_step

    cv2.line(
        frame,
        camera_tip,
        (camera_tip[0], map_top),
        (75, 75, 75),
        1,
        cv2.LINE_AA,
    )
    cv2.circle(frame, camera_tip, 6, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        "CAM",
        (camera_tip[0] - 14, map_bottom - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    def radar_point(forward, lateral):
        x = round(camera_tip[0] - lateral / MAP_HALF_WIDTH_METERS * map_half_width)
        y = round(map_bottom - forward / max_depth * depth_height)
        return (
            int(np.clip(x, map_left, map_right)),
            int(np.clip(y, map_top, map_bottom)),
        )

    robot_scale = min(
        map_half_width / MAP_HALF_WIDTH_METERS,
        depth_height / max_depth,
    )

    def robot_point(center, forward_offset, lateral_offset):
        x = round(center[0] - lateral_offset * robot_scale)
        y = round(center[1] - forward_offset * robot_scale)
        return (
            int(np.clip(x, map_left, map_right)),
            int(np.clip(y, map_top, map_bottom)),
        )

    for robot_index, bot in enumerate(robots):
        state = np.array([bot.x, bot.y, bot.yaw, bot.dyaw, bot.r], dtype=float)
        if not np.all(np.isfinite(state)) or bot.r <= 0:
            continue

        radius = float(bot.r)
        cos_yaw = np.cos(bot.yaw)
        sin_yaw = np.sin(bot.yaw)
        center = radar_point(bot.x, bot.y)
        local_corners = (
            (-radius, -radius),
            (radius, -radius),
            (radius, radius),
            (-radius, radius),
        )
        corners = []
        for forward_offset, lateral_offset in local_corners:
            rotated_forward = cos_yaw * forward_offset - sin_yaw * lateral_offset
            rotated_lateral = sin_yaw * forward_offset + cos_yaw * lateral_offset
            corners.append(robot_point(center, rotated_forward, rotated_lateral))

        cv2.polylines(
            frame,
            [np.asarray(corners, dtype=np.int32)],
            isClosed=True,
            color=(220, 220, 220),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        for plate_id in bot.plates:
            phase = bot.plate_phases.get(plate_id)
            if phase is None or not np.isfinite(phase):
                continue

            side_angle = bot.yaw + phase
            side_forward = -radius * np.cos(side_angle)
            side_lateral = -radius * np.sin(side_angle)
            tangent_forward = -np.sin(side_angle)
            tangent_lateral = np.cos(side_angle)
            side_start = robot_point(
                center,
                side_forward - radius * tangent_forward,
                side_lateral - radius * tangent_lateral,
            )
            side_end = robot_point(
                center,
                side_forward + radius * tangent_forward,
                side_lateral + radius * tangent_lateral,
            )
            side_index = int(round(phase / (np.pi / 2))) % len(PLATE_SIDE_COLORS)
            cv2.line(
                frame,
                side_start,
                side_end,
                PLATE_SIDE_COLORS[side_index],
                4,
                cv2.LINE_AA,
            )

        cv2.circle(frame, center, 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"R{robot_index + 1} {np.degrees(bot.dyaw):+.1f} deg/s",
            (center[0] - 25, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def track_video(model, video_path):
    video_path = Path(video_path)
    if not video_path.is_absolute():
        video_path = root / video_path

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.release()

    pose_tracking.plates.clear()
    pose_tracking.robotsnplates.clear()
    pose_tracking.PnPCalcs.clear()
    object_points = get_object_points()
    frame_count = 0
    next_plate_id = 0

    output_path = video_path.with_name(video_path.stem + "_pnp_overlay.mp4")
    writer = None
    start_time = time.perf_counter()
    results = model.predict(
        source=str(video_path),
        imgsz=imgsz,
        conf=0.1,
        tracker=str(root / "byteTrackConfig.yaml"),
        save=False,
        device=0,
        stream=True,
    )

    try:
        for result in results:
            frame_count += 1
            frame = result.orig_img.copy()
            frame_height, frame_width = frame.shape[:2]
            camera_matrix = get_camera_matrix(frame_width, frame_height)
            pose_tracking.PnPCalcs.clear()

            if result.keypoints is not None:
                current_plates, _, next_plate_id = pose_tracking.process_plates(
                    result,
                    camera_matrix,
                    object_points,
                    frame_count,
                    fps,
                    30,
                    next_plate_id,
                )
            else:
                current_plates = []

            current_robots = pose_tracking.update_robots(
                current_plates,
                frame_count,
                1.0 / fps,
            )
            pose_tracking.remove_expired_tracks(current_robots, frame_count)

            draw_robot_radar(frame, current_robots)
            if writer is None:
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (frame_width, frame_height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot create video: {output_path}")
            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()

    if frame_count == 0:
        raise RuntimeError(f"No frames were processed from: {video_path}")

    elapsed = time.perf_counter() - start_time
    print(f"Saved PnP visualization: {output_path}")
    if elapsed > 0.0:
        print(f"Processing speed: {frame_count / elapsed:.2f} FPS")
    return output_path


def main():
    model_path = root / "runs/detect/armor/armor_pose_EPOCH100-8/weights/best.pt"
    model = YOLO(str(model_path))
    for video in videos:
        track_video(model, video)


if __name__ == "__main__":
    main()
