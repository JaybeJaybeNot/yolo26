
from pathlib import Path
import time
import json
from ultralytics import YOLO
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import numpy as np

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
max_missing_frames = 1

# Approximate camera and large armor-plate geometry; replace with calibration data
# for metric-accurate pose estimates.
assumed_horizontal_fov_degrees = 60.0
plate_width_meters = 0.23
plate_height_meters = 0.127


def camera_matrix_for_frame(frame_shape):
    frame_height, frame_width = frame_shape[:2]
    focal_length = frame_width / (
        2.0 * np.tan(np.deg2rad(assumed_horizontal_fov_degrees) / 2.0)
    )
    return np.array([
        [focal_length, 0.0, frame_width / 2.0],
        [0.0, focal_length, frame_height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)


def pose_from_bounding_box(box, camera_matrix):
    x1, y1, x2, y2 = map(float, box)
    if x2 <= x1 or y2 <= y1:
        return None

    half_width = plate_width_meters / 2.0
    half_height = plate_height_meters / 2.0
    object_points = np.array([
        [-half_width, -half_height, 0.0],
        [half_width, -half_height, 0.0],
        [half_width, half_height, 0.0],
        [-half_width, half_height, 0.0],
    ], dtype=np.float32)
    image_points = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
    ], dtype=np.float32)

    solved, rotation_vector, translation = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        np.zeros((5, 1), dtype=float),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not solved or not np.isfinite(translation).all():
        return None

    rotation, _ = cv2.Rodrigues(rotation_vector)
    # Express the plate normal as the EKF's robot-heading angle.
    yaw = float(np.arctan2(rotation[2, 2], rotation[0, 2]))
    camera_x, camera_y, camera_depth = translation.reshape(3)
    if camera_depth <= 0.0:
        return None

    # EKF x/y are the horizontal plane; its z axis is vertical.
    return np.array(
        [camera_x, camera_depth, -camera_y, yaw],
        dtype=float,
    )


class robot:
    def __init__(
        self, x, y, robot_id, frame_id, sidex, sidey, color=None, z=0.0, yaw=0.0
    ):
        self.x = float(x)
        self.y = float(y)
        self.dx = 0.0
        self.dy = 0.0
        self.z = float(z)
        self.dz = 0.0
        self.yaw = float(yaw)
        self.dyaw = 0.0
        self.r = 0.0
        self.color = color
        self.id = robot_id
        self.sidex = float(sidex)
        self.sidey = float(sidey)
        self.frame_id = frame_id
        self.last_yolo_frame = frame_id
        self.last_detection_frame = frame_id

        # Initial EKF state and covariance
        self.x_prev = np.array(self.stateVector(), dtype=float)
        self.x_pred = self.x_prev.copy()
        self.P = np.eye(9)

        # Process noise parameters (acceleration noise)
        self.process_noise = [1.0, 1.0, 0.5, 1.0]
        self.radius_noise = 1e-5

        self.measurement_std = [0.15, 0.15, 0.15, 0.2]
        self.image_x = float(x)
        self.image_y = float(y)

    def move(self, dx, dy):
        self.x += dx
        self.y += dy
        self.x_prev = np.array(self.stateVector(), dtype=float)

    def stateVector(self):
        return [self.x, self.dx,self.y,self.dy,self.z,self.dz,self.yaw,self.dyaw,self.r]

    def stateTransition(self, x, dt):
        return np.array([
            x[0] + x[1] * dt,
            x[1],
            x[2] + x[3] * dt,
            x[3],
            x[4] + x[5] * dt,
            x[5],
            x[6] + x[7] * dt,
            x[7],
            x[8]
        ])

    def jacobian(self, dt):
        F = np.eye(9) #creates 9x9 Identity matrix (diagonal 1's)
        F[0, 1] = dt
        F[2, 3] = dt
        F[4, 5] = dt
        F[6, 7] = dt

        return F

    def h(self, x):
        return np.array([
            x[0] - x[8] * np.cos(x[6]),  # x_a
            x[2] - x[8] * np.sin(x[6]),  # y_a
            x[4],                        # z_a
            x[6]                         # yaw
        ])

    def jacobianH(self, x):
        H = np.zeros((4, 9))
        yaw = x[6]
        r = x[8]

        H[0, 0] = 1
        H[0, 6] = r * np.sin(yaw)
        H[0, 8] = -np.cos(yaw)

        H[1, 2] = 1
        H[1, 6] = -r * np.cos(yaw)
        H[1, 8] = -np.sin(yaw)

        H[2, 4] = 1
        H[3, 6] = 1

        return H

    def update_Q(self, dt):
        Q = np.zeros((9, 9))

        # Constant-velocity model with acceleration noise
        for i, q in zip([0, 2, 4, 6], self.process_noise):
            Q[i, i] = q * dt**4 / 4
            Q[i, i+1] = q * dt**3 / 2
            Q[i+1, i] = q * dt**3 / 2
            Q[i+1, i+1] = q * dt**2

        # Small radius uncertainty
        Q[8, 8] = self.radius_noise * dt

        return Q

    def update_R(self):
        # Measurement covariance: x, y, z, yaw
        return np.diag(np.square(self.measurement_std))

    def setState(self, x):
        self.x = float(x[0])
        self.dx = float(x[1])
        self.y = float(x[2])
        self.dy = float(x[3])
        self.z = float(x[4])
        self.dz = float(x[5])
        self.yaw = float(x[6])
        self.dyaw = float(x[7])
        self.r = float(max(0.0, x[8]))

        self.x_prev = np.array(self.stateVector(), dtype=float)

    def projected_plate_position(self, camera_matrix):
        plate_x, plate_y = self.plate_position()
        plate_z = self.h(np.array(self.stateVector(), dtype=float))[2]
        if plate_y <= 0.0:
            return self.image_x, self.image_y

        focal_x = camera_matrix[0, 0]
        focal_y = camera_matrix[1, 1]
        center_x = camera_matrix[0, 2]
        center_y = camera_matrix[1, 2]
        return (
            float(focal_x * plate_x / plate_y + center_x),
            float(center_y - focal_y * plate_z / plate_y),
        )

    def sync_image_position(self, camera_matrix):
        self.image_x, self.image_y = self.projected_plate_position(camera_matrix)

    def predict(self, dt, camera_matrix):
        F = self.jacobian(dt)

        self.x_pred = self.stateTransition(self.x_prev, dt)

        Q = self.update_Q(dt)

        self.P = F @ self.P @ F.T + Q

        self.x_prev = self.x_pred.copy()
        self.setState(self.x_prev)
        self.sync_image_position(camera_matrix)

        return self.x_pred

    def update(self, z):
        z = np.asarray(z, dtype=float)

        # Measurement covariance
        R = self.update_R()

        # Observation Jacobian
        H = self.jacobianH(self.x_pred)

        # Innovation covariance
        S = H @ self.P @ H.T + R

        # Kalman gain with singularity protection
        try:
            K = np.linalg.solve(S, H @ self.P).T
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.pinv(S)

        # Measurement residual
        innovation = z - self.h(self.x_pred)

        # Correct state estimate
        x_new = self.x_pred + K @ innovation

        # Joseph-form covariance update
        I = np.eye(9)
        A = I - K @ H

        self.P = A @ self.P @ A.T + K @ R @ K.T

        # Apply radius constraints and synchronize state
        self.setState(x_new)

        return np.array(self.stateVector())

    def plate_position(self):
        plate_state = self.h(np.array(self.stateVector(), dtype=float))
        return float(plate_state[0]), float(plate_state[1])
    #USEAGE example:
    #updated_state = botName.update(np.array([x,y,z,yaw]))


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
robots = {

}



def snapshot(frame_id):
    states = {}
    for robot_id, tracked_robot in robots.items():
        states[str(robot_id)] = {
            "id": tracked_robot.id,
            "frame_id": frame_id,
            "x": tracked_robot.image_x,
            "y": tracked_robot.image_y,
            "sidex": tracked_robot.sidex,
            "sidey": tracked_robot.sidey,
            "color": tracked_robot.color,
            "ekf_state": {
                "x": tracked_robot.x,
                "dx": tracked_robot.dx,
                "y": tracked_robot.y,
                "dy": tracked_robot.dy,
                "z": tracked_robot.z,
                "dz": tracked_robot.dz,
                "yaw": tracked_robot.yaw,
                "dyaw": tracked_robot.dyaw,
                "r": tracked_robot.r,
            },
            "last_yolo_frame": tracked_robot.last_yolo_frame,
            "last_detection_frame": tracked_robot.last_detection_frame,
        }
    return states

def save_overlay(video_path, json_path):
    history = json.loads(json_path.read_text(encoding="utf-8"))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = video_path.with_name(video_path.stem + "_overlay.mp4")
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create video: {output}")
    box_colors = {
        "red": (0, 0, 255),
        "blue": (255, 0, 0),
    }
    try:
        for positions in history:
            ok, frame = cap.read()
            if not ok:
                break
            for state in positions.values():
                color = box_colors.get(state["color"], (0, 255, 255))
                half_width = state["sidex"] / 2
                half_height = state["sidey"] / 2
                top_left = (
                    round(state["x"] - half_width),
                    round(state["y"] - half_height),
                )
                bottom_right = (
                    round(state["x"] + half_width),
                    round(state["y"] + half_height),
                )
                cv2.rectangle(frame, top_left, bottom_right, color, 2)
                ekf_state = state["ekf_state"]
                label = (
                    f"ID {state['id']} EKF "
                    f"x:{state['x']:.1f} y:{state['y']:.1f} "
                    f"vx:{ekf_state['dx']:.1f} vy:{ekf_state['dy']:.1f} "
                    f"yaw:{ekf_state['yaw']:.2f}"
                )
                cv2.putText(
                    frame, label,
                    (top_left[0], max(18, top_left[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
                )
            writer.write(frame)
    finally:
        cap.release()
        writer.release()
    print(f"Saved overlay: {output}")

def track_video(model, video_path, enemyColor, allyColor, maxPixDistance):
    video_path = Path(video_path)
    if not video_path.is_absolute():
        video_path = root / video_path
    robots.clear()
    next_robot_id = 0
    frame_count = 0  # Number of source frames represented in history
    yolo_calls = 0
    history = []
    start_time = time.perf_counter()
    executor = ThreadPoolExecutor(max_workers=1)
    pending = deque()

    def remove_expired_robots(current_frame):
        expired_ids = [
            robot_id
            for robot_id, tracked_robot in robots.items()
            if current_frame - tracked_robot.last_detection_frame
            > max_missing_frames * skip
        ]
        for robot_id in expired_ids:
            del robots[robot_id]

    def apply_grouping_results(frame_id, matches, frame_robot_ids):
        for plate_index, team, _ in matches:
            if team == "enemy":
                color = enemyColor.lower()
            elif team == "ally":
                color = allyColor.lower()
            else:
                raise ValueError(f"Unexpected grouping team: {team}")

            robot_id = frame_robot_ids[plate_index]
            if robot_id is None:
                continue
            tracked_robot = robots.get(robot_id)
            if tracked_robot is not None:
                tracked_robot.color = color

            robot_key = str(robot_id)
            for frame_state in history[frame_id:]:
                state = frame_state.get(robot_key)
                if state is not None:
                    state["color"] = color

    results = model.predict(
        source=str(video_path), imgsz=544, conf=0.2,
        save=False, device=0, stream=True, vid_stride=skip
    )
    try:
        for result in results:
            frame = result.orig_img
            camera_matrix = camera_matrix_for_frame(frame.shape)
            # Ultralytics normally yields source frames skip-1, 2*skip-1, ...
            source_frame = (yolo_calls + 1) * skip - 1
            while frame_count < source_frame:
                # Prediction for a skipped frame, without running YOLO
                for tracked_robot in robots.values():
                    tracked_robot.predict(1, camera_matrix)
                    tracked_robot.frame_id = frame_count
                remove_expired_robots(frame_count)
                history.append(snapshot(frame_count))
                frame_count += 1

            remove_expired_robots(source_frame)
            for tracked_robot in robots.values():
                dt = source_frame - tracked_robot.frame_id
                if dt > 0:
                    tracked_robot.predict(dt, camera_matrix)
                    tracked_robot.frame_id = source_frame

            plate_boxes = result.boxes.xyxy.cpu().numpy()
            xyxy_list = plate_boxes.tolist()

            matched_robot_ids = set()
            frame_robot_ids = []
            # Assign persistent IDs by nearest predicted position, without model.track().
            # Greedy one-to-one matching; threshold is in pixels.
            for xyxy in xyxy_list:
                xmin, ymin, xmax, ymax = xyxy
                sidex = xmax - xmin
                sidey = ymax - ymin
                center_x = (xmin + xmax) / 2
                center_y = (ymin + ymax) / 2
                pose = pose_from_bounding_box(xyxy, camera_matrix)
                if pose is None:
                    frame_robot_ids.append(None)
                    continue

                best_id = None
                best_dist = maxPixDistance ** 2
                for robot_id, tracked_robot in robots.items():
                    if robot_id in matched_robot_ids:
                        continue
                    pred_x, pred_y = tracked_robot.projected_plate_position(
                        camera_matrix
                    )
                    dist = (center_x - pred_x) ** 2 + (center_y - pred_y) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_id = robot_id

                if best_id is None:
                    best_id = next_robot_id
                    next_robot_id += 1
                    robots[best_id] = robot(
                        pose[0], pose[1], best_id, source_frame, sidex, sidey,
                        z=pose[2], yaw=pose[3]
                    )

                tracked_robot = robots[best_id]
                tracked_robot.update(pose)
                tracked_robot.sync_image_position(camera_matrix)
                matched_robot_ids.add(best_id)
                frame_robot_ids.append(best_id)
                tracked_robot.sidex = sidex
                tracked_robot.sidey = sidey
                tracked_robot.last_yolo_frame = source_frame
                tracked_robot.frame_id = source_frame
                tracked_robot.last_detection_frame = source_frame

            # Unmatched tracks still advance one frame to this YOLO frame.
            for robot_id, tracked_robot in robots.items():
                if robot_id not in matched_robot_ids:
                    tracked_robot.frame_id = source_frame
            history.append(snapshot(source_frame))
            frame_count = source_frame + 1
            yolo_calls += 1

            future = executor.submit(grouping, result.orig_img, plate_boxes,
                                     enemyColor, allyColor, maxPixDistance)
            pending.append((source_frame, future, frame_robot_ids))
            if len(pending) > 2:
                frame_id, completed, frame_robot_ids = pending.popleft()
                matches = completed.result()
                apply_grouping_results(frame_id, matches, frame_robot_ids)
                for _, team, center in matches:
                    print(f"Frame: {frame_id} | Team: {team} | Center: {center}")
        while pending:
            frame_id, completed, frame_robot_ids = pending.popleft()
            matches = completed.result()
            apply_grouping_results(frame_id, matches, frame_robot_ids)
            for _, team, center in matches:
                print(f"Frame: {frame_id} | Team: {team} | Center: {center}")
    finally:
        executor.shutdown(wait=True)
    elapsed = time.perf_counter() - start_time
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    while frame_count < total_frames:
        for tracked_robot in robots.values():
            tracked_robot.predict(1, camera_matrix)
            tracked_robot.frame_id = frame_count
        remove_expired_robots(frame_count)
        history.append(snapshot(frame_count))
        frame_count += 1
    print(f"Video: {video_path.name}")
    print(f"Total frames: {frame_count}")
    print(f"YOLO inference calls: {yolo_calls}")
    print(f"Predicted frames: {frame_count - yolo_calls}")
    print(f"Tracking time: {elapsed:.2f}s")
    if elapsed > 0:
        print(f"Effective processing FPS: {frame_count / elapsed:.2f}")
    json_path = video_path.with_name(video_path.stem + "_robots.json")
    json_path.write_text(json.dumps(history))
    print(f"Saved robot states: {json_path}")
    save_overlay(video_path, json_path)
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
        List of (plate_index, team, center) tuples.

        Example:
            [
                (0, "enemy", (320, 240)),
                (1, "ally", (510, 180))
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
            (plate_index, team, center)
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
