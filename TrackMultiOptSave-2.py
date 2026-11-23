
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

#TO DO FOR LATER MODEL:
#REIMPLEMENT KEYPOINT DETECTION BUT WITH COLOR

skip = 2

root = Path(__file__).resolve().parent

videos = [
    "dji_best_local.mp4",
    "[BLUR_TEST]RoboMasterTestVideo3.mp4",
    "RoboMasterTestVideo2.mp4",
]
#parameters for alpha beta filter
alpha = 0.3
beta = (alpha * alpha)/ (2- alpha)
gamma = 0.8
max_missing_frames = 1

imgsz = 544



class plate:
    def __init__(self, x, y, id, frame_id, sidex, sidey, color=None):
        self.x = float(x)
        self.y = float(y)

        self.sidex = sidex
        self.sidey = sidey

        self.color = color

        self.dx = 0.0
        self.dy = 0.0

        self.ddx = 0.0
        self.ddy = 0.0

        self.relativeX = 0.0
        self.relativeY = 0.0
        self.relativeZ = 0.0

        self.relativerX = 0.0
        self.relativerY = 0.0
        self.relativerZ = 0.0

        self.id = id

        # Last position actually observed by YOLO
        self.last_yolo_frame = frame_id
        self.frame_id = frame_id
        self.last_yolo_x = float(x)
        self.last_yolo_y = float(y)
        self.last_detection_frame = frame_id

    def updateRelative(self, R, t):
        self.relativeX = R[0]
        self.relativeY = R[1]
        self.relativeZ = R[2]
        self.relativerX = t[0]
        self.relativerY = t[1]
        self.relativerZ = t[2]

    def move(self, frame_count):
        dt = frame_count - self.last_yolo_frame

        # 1. Prediction step (includes acceleration)
        pred_x = self.x + self.dx * dt + 0.5 * self.ddx * (dt ** 2)
        pred_y = self.y + self.dy * dt + 0.5 * self.ddy * (dt ** 2)

        pred_dx = self.dx + self.ddx * dt
        pred_dy = self.dy + self.ddy * dt

        # 2. Residual calculation (measurement - prediction)
        rx = self.last_yolo_x - pred_x
        ry = self.last_yolo_y - pred_y

        # 3. State update step
        if dt > 0:
            self.ddx += (2 * gamma * rx) / (dt ** 2)
            self.ddy += (2 * gamma * ry) / (dt ** 2)

            self.dx = pred_dx + (beta * rx) / dt
            self.dy = pred_dy + (beta * ry) / dt
        else:
            self.dx = pred_dx
            self.dy = pred_dy

        self.x = pred_x + alpha * rx
        self.y = pred_y + alpha * ry

        self.frame_id = frame_count
        self.last_yolo_frame = frame_count

    def reset(self, x, y):
        self.x = x
        self.y = y



class robot:
    def __init__(self,x,y,z,dx,dy,dz,yaw,dyaw,r,color):
        # Rotation center in odom coordinates
        self.x = float(x)
        self.y = float(y)

        # Translational velocity
        self.dx = float(dx)
        self.dy = float(dy)

        # Current armor plate height and vertical velocity
        self.z = float(z)
        self.dz = float(dz)

        # Vehicle rotation (radians and radians/second)
        self.yaw = float(yaw)
        self.dyaw = float(dyaw)

        # Horizontal distance from armor plate to rotation center
        self.r = float(r)

        # Team color and associated plates
        self.color = color
        self.plates = {}

        # Initial EKF state and covariance
        self.x_prev = np.array(self.stateVector(), dtype=float)
        self.x_pred = self.x_prev.copy()
        self.P = np.eye(9)

        # Process noise parameters (acceleration noise)
        self.process_noise = [1.0, 1.0, 0.5, 1.0]
        self.radius_noise = 1e-5

        # Measurement standard deviations: x, y, z, yaw
        self.measurement_std = [0.03, 0.03, 0.03, 0.05]

        

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

    def update_R(self, z):
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
        self.r = float(np.clip(x[8], 0.12, 0.35))

        self.x_prev = np.array(self.stateVector(), dtype=float)

    def predict(self, dt):
        F = self.jacobian(dt)

        self.x_pred = self.stateTransition(self.x_prev, dt)

        Q = self.update_Q(dt)

        self.P = F @ self.P @ F.T + Q

        self.x_prev = self.x_pred.copy()
        self.setState(self.x_prev)

        return self.x_pred

    def update(self, z):
        z = np.asarray(z, dtype=float)

        # Measurement covariance
        R = self.update_R(z)

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
#set contain plates
plates = {

}

robots = {
    
}

def get_camera_matrix(width, height, hfov=70):

    f = width / (2 * np.tan(np.radians(hfov / 2)))

    return np.array([
        [f, 0, width / 2],
        [0, f, height / 2],
        [0, 0, 1]
    ], dtype=np.float64)
K = get_camera_matrix(imgsz, imgsz)
dist_coeffs = np.zeros(5, dtype=np.float64)

#for PnP. Measurement may change later. 
SMALL_ARMOR_WIDTH = 0.135
LARGE_ARMOR_WIDTH = 0.230
ARMOR_HEIGHT = 0.055
def get_object_points(is_large=False):
    w = LARGE_ARMOR_WIDTH if is_large else SMALL_ARMOR_WIDTH
    h = ARMOR_HEIGHT

    return np.array([
        [-w/2, -h/2, 0],  # Top left
        [-w/2,  h/2, 0],  # Bottom left
        [ w/2,  h/2, 0],  # Bottom right
        [ w/2, -h/2, 0]   # Top right
    ], dtype=np.float64)

# INPUTS:
# image_points: 2D pixel coordinates of four keypoints (4x2)
# K: Camera intrinsic matrix (3x3)
# dist_coeffs: Camera lens distortion coefficients
# object_points: Known 3D coordinates of four keypoints (4x3) [THIS IS get_object_points]
#
# OUTPUTS:
# R: Object-to-camera rotation matrix (3x3) 
# t: Object origin's position relative to the camera [x, y, z]
# None: If pose estimation fails

def PnP(image_points, K, dist_coeffs, object_points):
    image_points = np.array(
        image_points, dtype=np.float64
    ).reshape(4, 2)

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        K,
        dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP
    )

    if not success:
        return None

    R, _ = cv2.Rodrigues(rvec)

    return R, tvec.flatten()



def snapshot(frame_id):
    return {str(pid): {"id": p.id, "frame_id": frame_id,
            "x": p.x, "y": p.y, "dx": p.dx, "dy": p.dy,
            "sidex": p.sidex, "sidey": p.sidey, "color": p.color,
            "last_yolo_frame": p.last_yolo_frame,
            "last_detection_frame": p.last_detection_frame}
            for pid, p in plates.items()}

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
                color = box_colors.get(state["color"])
                if color is None:
                    continue
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
            writer.write(frame)
    finally:
        cap.release()
        writer.release()
    print(f"Saved overlay: {output}")

def track_video(model, video_path, enemyColor, allyColor, maxPixDistance):
    video_path = Path(video_path)
    if not video_path.is_absolute():
        video_path = root / video_path
    plates.clear()
    next_plate_id = 0
    frame_count = 0  # Number of source frames represented in history
    yolo_calls = 0
    history = []
    start_time = time.perf_counter()
    executor = ThreadPoolExecutor(max_workers=1)
    pending = deque()

    def remove_expired_plates(current_frame):
        expired_ids = [
            plate_id
            for plate_id, p in plates.items()
            if current_frame - p.last_detection_frame > max_missing_frames
        ]
        for plate_id in expired_ids:
            del plates[plate_id]

    def apply_grouping_results(frame_id, matches, frame_plate_ids):
        for plate_index, team, _ in matches:
            if team == "enemy":
                color = enemyColor.lower()
            elif team == "ally":
                color = allyColor.lower()
            else:
                raise ValueError(f"Unexpected grouping team: {team}")

            plate_id = frame_plate_ids[plate_index]
            tracked_plate = plates.get(plate_id)
            if tracked_plate is not None:
                tracked_plate.color = color

            plate_key = str(plate_id)
            for frame_state in history[frame_id:]:
                state = frame_state.get(plate_key)
                if state is not None:
                    state["color"] = color

    results = model.predict(
        source=str(video_path), imgsz=imgsz, conf=0.2,
        save=False, device=0, stream=True, vid_stride=skip
    )
    try:
        for result in results:
            # Ultralytics normally yields source frames skip-1, 2*skip-1, ...
            source_frame = (yolo_calls + 1) * skip - 1
            while frame_count < source_frame:
                # Prediction for a skipped frame, without running YOLO
                for p in plates.values():
                    p.move(frame_count)
                    p.frame_id = frame_count
                remove_expired_plates(frame_count)
                history.append(snapshot(frame_count))
                frame_count += 1

            remove_expired_plates(source_frame)

            plate_boxes = result.boxes.xyxy.cpu().numpy()
            xyxy_list = plate_boxes.tolist()

            matched_ids = set()
            frame_plate_ids = []
            # Assign persistent IDs by nearest predicted position, without model.track().
            # Greedy one-to-one matching; threshold is in pixels.
            for xyxy in xyxy_list:
                xmin, ymin, xmax, ymax = xyxy
                sidex = xmax - xmin
                sidey = ymax - ymin
                center_x = (xmin + xmax) / 2
                center_y = (ymin + ymax) / 2
                best_id = None
                best_dist = maxPixDistance ** 2
                for pid, p in plates.items():
                    if pid in matched_ids:
                        continue
                    # p.x/y already advanced through preceding skipped frames.
                    pred_x = p.x + p.dx
                    pred_y = p.y + p.dy
                    dist = (center_x - pred_x) ** 2 + (center_y - pred_y) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_id = pid

                if best_id is None:
                    best_id = next_plate_id
                    next_plate_id += 1
                    plates[best_id] = plate(
                        center_x, center_y, best_id, source_frame, sidex, sidey
                    )
                matched_ids.add(best_id)
                frame_plate_ids.append(best_id)
                p = plates[best_id]
                dt = source_frame - p.last_yolo_frame
                # Use the current predicted state for residuals; skipped frames
                # have already advanced p.x/y, so do not extrapolate dt twice.
                pred_x = p.x + (p.dx if dt > 0 else 0)
                pred_y = p.y + (p.dy if dt > 0 else 0)
                rx = center_x - pred_x
                ry = center_y - pred_y
                if dt > 0:
                    p.dx += beta * rx / dt
                    p.dy += beta * ry / dt
                p.x = pred_x + alpha * rx
                p.y = pred_y + alpha * ry
                p.sidex = sidex
                p.sidey = sidey

                p.last_yolo_frame = source_frame
                p.frame_id = source_frame
                p.last_detection_frame = source_frame

                # ---------- PnP Pose estimation ----------

                #image_points = extract_keypoints(result.orig_img, xyxy)
                image_points = extract_keypoints(result.orig_img, xyxy)

                if image_points is not None:
                    pose = PnP(image_points, K, dist_coeffs, object_points)

                    if pose is not None:
                        p.R, p.t = pose

            # Unmatched tracks still advance one frame to this YOLO frame.
            for pid, p in list(plates.items()):
                if pid not in matched_ids:
                    p.move(frame_count)
                    p.frame_id = source_frame
            history.append(snapshot(source_frame))
            frame_count = source_frame + 1
            yolo_calls += 1

            future = executor.submit(grouping, result.orig_img, plate_boxes,
                                     enemyColor, allyColor, maxPixDistance)
            pending.append((source_frame, future, frame_plate_ids))
            if len(pending) > 2:
                frame_id, completed, frame_plate_ids = pending.popleft()
                matches = completed.result()
                apply_grouping_results(frame_id, matches, frame_plate_ids)
                for _, team, center in matches:
                    print(f"Frame: {frame_id} | Team: {team} | Center: {center}")
        while pending:
            frame_id, completed, frame_plate_ids = pending.popleft()
            matches = completed.result()
            apply_grouping_results(frame_id, matches, frame_plate_ids)
            for _, team, center in matches:
                print(f"Frame: {frame_id} | Team: {team} | Center: {center}")
    finally:
        executor.shutdown(wait=True)
    elapsed = time.perf_counter() - start_time
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    while frame_count < total_frames:
        for p in plates.values():
            p.move(frame_count)
            p.frame_id = frame_count
        remove_expired_plates(frame_count)
        history.append(snapshot(frame_count))
        frame_count += 1
    print(f"Video: {video_path.name}")
    print(f"Total frames: {frame_count}")
    print(f"YOLO inference calls: {yolo_calls}")
    print(f"Predicted frames: {frame_count - yolo_calls}")
    print(f"Tracking time: {elapsed:.2f}s")
    if elapsed > 0:
        print(f"Effective processing FPS: {frame_count / elapsed:.2f}")
    json_path = video_path.with_name(video_path.stem + "_plates.json")
    json_path.write_text(json.dumps(history))
    print(f"Saved plate states: {json_path}")
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
