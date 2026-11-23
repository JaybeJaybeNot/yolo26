from pathlib import Path
import time
from ultralytics import YOLO
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import math

imgsz = 544
skip = 2
root = Path(__file__).resolve().parent

videos = [
    #"FirstPerson1.mp4",
    #"1.mp4",
    "dji_best_local.mp4",
    #"[BLUR_TEST]RoboMasterTestVideo3.mp4",
    #"RoboMasterTestVideo2.mp4",
]

alpha = 0.3
beta = (alpha * alpha) / (2 - alpha)
gamma = 0.8
max_missing_frames = 1

MIN_KEYPOINT_CONF = 0.75
MAX_REPROJECTION_ERROR = 1.5
POSE_AMBIGUITY_THRESHOLD = 0.2

MAX_PLATE_DISTANCE = 0.7
MAX_CENTER_DISTANCE = 0.35
MAX_SPEED_DIFFERENCE = 1.0
MAX_ROBOT_DISTANCE = 0.7
INITIAL_RADIUS = 0.23

CAMERA_TO_ROBOT = np.array([
    [0, 0, 1],
    [-1, 0, 0],
    [0, -1, 0]
], dtype=float)

#basic implementation for calculating angle to shoot. Don't predict with angular speed yet -- too volatile. 
#assume v1 = 30 m/s
#outputs: horizontal angle, vertical angle.
def calculateTrajectory(x,y,z):
    v1 = 30.0  # muzzle velocity in m/s
    g = 9.81   # acceleration due to gravity in m/s^2
    # 1. Horizontal angle (yaw) in radians
    horizontal_angle = math.atan2(y, x)
    # 2. Horizontal ground distance to target
    r = math.hypot(x, y)
    # Edge case: target is directly above or below the turret
    if r == 0:
        if z > 0 and (v1**2 >= 2 * g * z):
            return horizontal_angle, math.pi / 2
        elif z < 0:
            return horizontal_angle, -math.pi / 2
        else:
            return None  # Target unreachable

    # 3. Discriminant from the projectile trajectory equation
    discriminant = v1**4 - g * (g * r**2 + 2 * z * v1**2)
    # If discriminant < 0, the target is out of range for v1 = 30 m/s
    if discriminant < 0:
        return None
    # 4. Vertical angle (pitch) in radians
    # Using the minus sign gives the direct (low-arc) trajectory
    vertical_angle = math.atan((v1**2 - math.sqrt(discriminant)) / (g * r))
    return horizontal_angle, vertical_angle


class plate:
    def __init__(self, keypoints, id, frame_id, sidex, sidey, color=None):
        self.keypoints = np.array(keypoints, dtype=float)
        self.sidex = sidex
        self.sidey = sidey
        self.color = color

        self.dkeypoints = np.zeros((4, 2))
        self.ddkeypoints = np.zeros((4, 2))

        self.relativeX = 0.0
        self.relativeY = 0.0
        self.relativeZ = 0.0

        self.relativerX = 0.0
        self.relativerY = 0.0
        self.relativerZ = 0.0

        self.position = None
        self.velocity = np.zeros(3)
        self.center_position = None
        self.center_velocity = np.zeros(3)
        self.velocity_valid = False
        self.yaw = 0.0

        self.R = None
        self.t = None

        self.id = id
        self.last_yolo_frame = frame_id
        self.frame_id = frame_id
        self.last_yolo_keypoints = self.keypoints.copy()
        self.last_detection_frame = frame_id

    def updateRelative(self, R, t):
        self.relativeX = float(t[0])
        self.relativeY = float(t[1])
        self.relativeZ = float(t[2])

        # Euler angles in degrees
        angles = cv2.RQDecomp3x3(R)[0]
        self.relativerX = float(angles[0])
        self.relativerY = float(angles[1])
        self.relativerZ = float(angles[2])

    def updateMotion(self, R, t, frame_count, fps):
        position = CAMERA_TO_ROBOT @ t
        normal = CAMERA_TO_ROBOT @ R[:, 2]

        yaw = np.arctan2(normal[1], normal[0])

        center_position = position + INITIAL_RADIUS * np.array([
            np.cos(yaw),
            np.sin(yaw),
            0.0
        ])

        dt = (frame_count - self.last_detection_frame) / fps

        if self.position is not None and dt > 0:
            velocity = (position - self.position) / dt
            center_velocity = (center_position - self.center_position) / dt

            if self.velocity_valid:
                self.velocity = 0.7 * self.velocity + 0.3 * velocity
                self.center_velocity = 0.7 * self.center_velocity + 0.3 * center_velocity
            else:
                self.velocity = velocity
                self.center_velocity = center_velocity

            self.velocity_valid = True

        self.position = position
        self.center_position = center_position
        self.yaw = yaw
        self.R = R.copy()
        self.t = t.copy()

        self.updateRelative(R, t)
        self.last_detection_frame = frame_count
        self.frame_id = frame_count

    def measurement(self):
        return np.array([
            self.position[0],
            self.position[1],
            self.position[2],
            self.yaw
        ])

    def move(self, frame_count):
        dt = frame_count - self.last_yolo_frame

        pred_keypoints = self.keypoints + self.dkeypoints * dt + 0.5 * self.ddkeypoints * (dt ** 2)
        pred_dkeypoints = self.dkeypoints + self.ddkeypoints * dt

        r = self.last_yolo_keypoints - pred_keypoints

        if dt > 0:
            self.ddkeypoints += (2 * gamma * r) / (dt ** 2)
            self.dkeypoints = pred_dkeypoints + (beta * r) / dt
        else:
            self.dkeypoints = pred_dkeypoints

        self.keypoints = pred_keypoints + alpha * r

        self.frame_id = frame_count
        self.last_yolo_frame = frame_count

    def reset(self, keypoints):
        self.keypoints = np.array(keypoints, dtype=float)


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
            min(frame_height, int(self.y + self.height / 2 + self.margin))
        )


class robot:
    def __init__(self,x,y,z,dx,dy,dz,yaw,dyaw,r,color):
        # Rotation center
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
        self.plate_phases = {}
        self.last_detection_frame = 0

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
        F = np.eye(9)
        F[0, 1] = dt
        F[2, 3] = dt
        F[4, 5] = dt
        F[6, 7] = dt

        return F

    def h(self, x, phase=0.0):
        yaw = x[6] + phase

        return np.array([
            x[0] - x[8] * np.cos(yaw),
            x[2] - x[8] * np.sin(yaw),
            x[4],
            yaw
        ])

    def jacobianH(self, x, phase=0.0):
        H = np.zeros((4, 9))
        yaw = x[6] + phase
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

    def update(self, z, phase=0.0):
        z = np.asarray(z, dtype=float)

        # Measurement covariance
        R = self.update_R(z)

        # Observation Jacobian
        H = self.jacobianH(self.x_pred, phase)

        # Innovation covariance
        S = H @ self.P @ H.T + R

        # Kalman gain with singularity protection
        try:
            K = np.linalg.solve(S, H @ self.P).T
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.pinv(S)

        # Measurement residual
        innovation = z - self.h(self.x_pred, phase)
        innovation[3] = np.arctan2(
            np.sin(innovation[3]),
            np.cos(innovation[3])
        )

        # Correct state estimate
        x_new = self.x_pred + K @ innovation

        # Joseph-form covariance update
        I = np.eye(9)
        A = I - K @ H
        self.P = A @ self.P @ A.T + K @ R @ K.T

        # Apply radius constraints and synchronize state
        self.setState(x_new)
        self.x_pred = np.array(self.stateVector(), dtype=float)

        return self.x_pred

    def updatePlate(self, p):
        # Determine which side of the robot the plate belongs to
        if p.id not in self.plate_phases:
            difference = p.yaw - self.yaw
            difference = np.arctan2(np.sin(difference), np.cos(difference))

            phase = np.round(difference / (np.pi / 2)) * (np.pi / 2)
            self.plate_phases[p.id] = phase

        self.plates[p.id] = p

        return self.update(
            p.measurement(),
            self.plate_phases[p.id]
        )


tracks = []
plates = {}
PnPCalcs = []
robotsnplates = []


def compatible_plates(p1, p2):
    if p1.color is not None and p2.color is not None:
        if p1.color != p2.color:
            return False

    distance = np.linalg.norm(p1.position - p2.position)

    if distance > MAX_PLATE_DISTANCE:
        return False

    center_distance = np.linalg.norm(
        p1.center_position[:2] - p2.center_position[:2]
    )

    if center_distance > MAX_CENTER_DISTANCE:
        return False

    if p1.velocity_valid and p2.velocity_valid:
        speed_difference = np.linalg.norm(
            p1.center_velocity - p2.center_velocity
        )

        if speed_difference > MAX_SPEED_DIFFERENCE:
            return False

    return True


def group_plates(current_plates):
    groups = []

    for p in current_plates:
        compatible_groups = []

        for group in groups:
            if all(compatible_plates(p, member) for member in group):
                center = np.mean(
                    [member.center_position for member in group],
                    axis=0
                )

                distance = np.linalg.norm(p.center_position - center)
                compatible_groups.append((distance, group))

        if compatible_groups:
            _, group = min(
                compatible_groups,
                key=lambda candidate: candidate[0]
            )
            group.add(p)
        else:
            groups.append({p})

    return groups


def group_velocity(group):
    velocities = [
        p.center_velocity
        for p in group
        if p.velocity_valid
    ]

    if not velocities:
        return None

    return np.mean(velocities, axis=0)


def update_robots(current_plates, frame_count, dt):
    groups = group_plates(current_plates)

    # Predict all existing robot states
    for bot in robotsnplates:
        bot.predict(dt)

    group_centers = [
        np.mean([p.center_position for p in group], axis=0)
        for group in groups
    ]

    group_velocities = [
        group_velocity(group)
        for group in groups
    ]

    candidates = []

    # Associate groups with existing robots
    for i, group in enumerate(groups):
        ids = {p.id for p in group}
        center = group_centers[i]
        velocity = group_velocities[i]

        for j, bot in enumerate(robotsnplates):
            overlap = len(ids.intersection(bot.plates.keys()))

            robot_center = np.array([bot.x, bot.y, bot.z])
            distance = np.linalg.norm(center - robot_center)

            if overlap == 0:
                if distance > MAX_ROBOT_DISTANCE:
                    continue

                if velocity is not None:
                    robot_velocity = np.array([
                        bot.dx,
                        bot.dy,
                        bot.dz
                    ])

                    speed_difference = np.linalg.norm(
                        velocity - robot_velocity
                    )

                    if speed_difference > MAX_SPEED_DIFFERENCE:
                        continue

            candidates.append((-overlap, distance, i, j))

    matched_groups = set()
    matched_robots = set()

    # Prioritize shared plate IDs, then spatial proximity
    for _, _, i, j in sorted(candidates):
        if i in matched_groups or j in matched_robots:
            continue

        bot = robotsnplates[j]

        bot.plates = {}

        for p in sorted(groups[i], key=lambda p: p.id):
            bot.updatePlate(p)

        bot.last_detection_frame = frame_count

        matched_groups.add(i)
        matched_robots.add(j)

    # Initialize robots for unmatched plate groups
    for i, group in enumerate(groups):
        if i in matched_groups:
            continue

        center = group_centers[i]
        velocity = group_velocities[i]

        if velocity is None:
            velocity = np.zeros(3)

        first_plate = min(group, key=lambda p: p.id)

        bot = robot(
            center[0],
            center[1],
            center[2],
            velocity[0],
            velocity[1],
            velocity[2],
            first_plate.yaw,
            0.0,
            INITIAL_RADIUS,
            first_plate.color
        )

        bot.plate_phases[first_plate.id] = 0.0
        bot.plates[first_plate.id] = first_plate
        bot.last_detection_frame = frame_count

        for p in sorted(group, key=lambda p: p.id):
            if p.id != first_plate.id:
                bot.updatePlate(p)

        robotsnplates.append(bot)

    # Remove robots that have not been observed recently
    robotsnplates[:] = [
        bot for bot in robotsnplates
        if frame_count - bot.last_detection_frame <= max_missing_frames
    ]

    return robotsnplates


def get_camera_matrix(width, height, hfov=70):
    f = width / (2 * np.tan(np.radians(hfov / 2)))
    return np.array([
        [f, 0, width / 2],
        [0, f, height / 2],
        [0, 0, 1]
    ], dtype=np.float64)


dist_coeffs = np.zeros(5, dtype=np.float64)

SMALL_ARMOR_WIDTH = 0.135
LARGE_ARMOR_WIDTH = 0.230
ARMOR_HEIGHT = 0.055


def get_object_points(is_large=False):
    w = LARGE_ARMOR_WIDTH if is_large else SMALL_ARMOR_WIDTH
    h = ARMOR_HEIGHT

    return np.array([
        [-w / 2, -h / 2, 0],  # Top left
        [-w / 2,  h / 2, 0],  # Bottom left
        [ w / 2,  h / 2, 0],  # Bottom right
        [ w / 2, -h / 2, 0]   # Top right
    ], dtype=np.float64)


def reprojection_error(
    object_points, image_points, K, dist_coeffs, rvec, tvec
):
    projected_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        K,
        dist_coeffs
    )

    difference = projected_points.reshape(-1, 2) - image_points

    return float(np.sqrt(np.mean(np.sum(difference ** 2, axis=1))))


def pose_difference(R, t, previous_pose):
    previous_R, previous_t = previous_pose
    rotation_difference = previous_R.T @ R

    angle = np.arccos(
        np.clip(
            (np.trace(rotation_difference) - 1) / 2,
            -1.0,
            1.0
        )
    )

    translation_difference = (
        np.linalg.norm(t - previous_t)
        / max(np.linalg.norm(previous_t), 1e-6)
    )

    return angle + translation_difference

def refine_pose_lm(
    object_points,
    image_points,
    K,
    dist_coeffs,
    rvec,
    tvec,
    max_iters=20,
    lambda_init=1e-3,
    tol=1e-10,
):
    """Levenberg-Marquardt refinement of a pose by minimizing reprojection error."""
    obj = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64)
    dist = (
        np.zeros(5)
        if dist_coeffs is None
        else np.asarray(dist_coeffs, dtype=np.float64)
    )

    # State: [rx, ry, rz, tx, ty, tz]
    x = np.concatenate([np.ravel(rvec), np.ravel(tvec)]).astype(np.float64)

    def evaluate(x):
        proj, jac = cv2.projectPoints(obj, x[:3], x[3:], K, dist)
        residual = (proj.reshape(-1, 2) - img).ravel()   # (2N,)
        # jac columns: rvec(3), tvec(3), f(2), c(2), dist(...) -> keep pose only
        return residual, jac[:, :6]                      # (2N, 6)

    r, J = evaluate(x)
    cost = r @ r
    lam = lambda_init

    for _ in range(max_iters):
        converged = False
        A = J.T @ J
        g = J.T @ r

        improved = False
        for _ in range(10):  # damping search
            # Marquardt scaling: damp proportionally to each diagonal term
            H = A + lam * np.diag(np.maximum(np.diag(A), 1e-12))
            try:
                dx = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue

            x_new = x + dx
            r_new, J_new = evaluate(x_new)
            cost_new = r_new @ r_new

            if np.isfinite(cost_new) and cost_new < cost:
                # Step accepted: trust the model more (toward Gauss-Newton)
                converged = (cost - cost_new) < tol * max(cost, 1e-12)
                x, r, J, cost = x_new, r_new, J_new, cost_new
                lam = max(lam / 10.0, 1e-12)
                improved = True
                break

            # Step rejected: trust the model less (toward gradient descent)
            lam *= 10.0

        if not improved or converged:
            break

    return x[:3].reshape(3, 1), x[3:].reshape(3, 1)


def PnP(
    image_points,
    K,
    dist_coeffs,
    object_points,
    keypoint_conf=None,
    previous_pose=None
):
    image_points = np.array(
        image_points, dtype=np.float64
    ).reshape(4, 2)
    object_points = np.array(
        object_points, dtype=np.float64
    ).reshape(4, 3)

    # Reject invalid or low-confidence keypoints
    if not np.all(np.isfinite(image_points)):
        return None

    if keypoint_conf is not None:
        if not np.all(np.isfinite(keypoint_conf)):
            return None
        if np.any(keypoint_conf < MIN_KEYPOINT_CONF):
            return None

    # Reject degenerate or incorrectly intersecting quadrilaterals
    if not cv2.isContourConvex(image_points.astype(np.float32)):
        return None

    # 1) Closed-form initial poses from IPPE (up to 2 candidates)
    success, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        object_points,
        image_points,
        K,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE
    )

    if not success:
        return None

    candidates = []

    # 2) Refine each IPPE candidate with the custom LM
    for rvec, tvec in zip(rvecs, tvecs):
        if not np.all(np.isfinite(rvec)):
            continue

        if not np.all(np.isfinite(tvec)):
            continue

        rvec, tvec = refine_pose_lm(
            object_points,
            image_points,
            K,
            dist_coeffs,
            rvec,
            tvec
        )

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.flatten()

        if not np.all(np.isfinite(R)):
            continue

        if not np.all(np.isfinite(t)):
            continue

        # All object points must be in front of the camera
        camera_points = (R @ object_points.T).T + t

        if np.any(camera_points[:, 2] <= 0):
            continue

        error = reprojection_error(
            object_points,
            image_points,
            K,
            dist_coeffs,
            rvec,
            tvec
        )

        if not np.isfinite(error):
            continue

        if error > MAX_REPROJECTION_ERROR:
            continue

        candidates.append((error, R, t))

    if not candidates:
        return None

    candidates.sort(key=lambda candidate: candidate[0])
    best_error, best_R, best_t = candidates[0]

    # 3) Resolve ambiguity using the previous frame's pose
    if previous_pose is not None:
        ambiguous_candidates = [
            candidate for candidate in candidates
            if candidate[0] <= best_error + POSE_AMBIGUITY_THRESHOLD
        ]

        _, best_R, best_t = min(
            ambiguous_candidates,
            key=lambda candidate: pose_difference(
                candidate[1],
                candidate[2],
                previous_pose
            )
        )

    return best_R, best_t


def get_video_fps(video_path):
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()

    return fps if np.isfinite(fps) and fps > 0 else 30.0


def extract_detections(result):
    plate_boxes = result.boxes.xyxy.cpu().numpy()
    plate_keypoints = result.keypoints.xy.cpu().numpy()

    keypoint_confidences = (
        result.keypoints.conf.cpu().numpy()
        if result.keypoints.conf is not None
        else None
    )

    return plate_boxes, plate_keypoints, keypoint_confidences


def match_previous_plate(image_points, previous_plates, matched_previous, maxPixDistance):
    center = np.mean(image_points, axis=0)
    previous_plate = None
    minimum_distance = maxPixDistance

    for p in previous_plates:
        if p.id in matched_previous:
            continue

        previous_center = np.mean(p.keypoints, axis=0)
        distance = np.linalg.norm(center - previous_center)

        if distance < minimum_distance:
            minimum_distance = distance
            previous_plate = p

    return previous_plate


def process_plate(image_points, plate_box, keypoint_conf, previous_plate,
                  K, object_points, frame_count, fps, next_plate_id):

    previous_pose = (
        (previous_plate.R, previous_plate.t)
        if previous_plate is not None
        else None
    )

    pose = PnP(
        image_points, K, dist_coeffs,
        object_points, keypoint_conf, previous_pose
    )

    if pose is None:
        return None, next_plate_id

    R, t = pose
    x1, y1, x2, y2 = plate_box

    if previous_plate is not None:
        p = previous_plate
        p.reset(image_points)
        p.sidex = x2 - x1
        p.sidey = y2 - y1
        p.last_yolo_keypoints = p.keypoints.copy()
        p.last_yolo_frame = frame_count

    else:
        p = plate(
            image_points, next_plate_id, frame_count,
            x2 - x1, y2 - y1
        )

        plates[next_plate_id] = p
        next_plate_id += 1

    p.updateMotion(R, t, frame_count, fps)
    PnPCalcs.append((R, t))

    print(f"Position: {t}")

    return p, next_plate_id


def process_plates(result, K, object_points, frame_count,
                   fps, maxPixDistance, next_plate_id):

    plate_boxes, plate_keypoints, keypoint_confidences = extract_detections(result)

    previous_plates = [
        p for p in plates.values()
        if frame_count - p.last_detection_frame <= max_missing_frames + 1
    ]

    matched_previous = set()
    current_plates = []

    for i, image_points in enumerate(plate_keypoints):
        keypoint_conf = (
            keypoint_confidences[i]
            if keypoint_confidences is not None
            else None
        )

        previous_plate = match_previous_plate(
            image_points, previous_plates,
            matched_previous, maxPixDistance
        )

        p, next_plate_id = process_plate(
            image_points, plate_boxes[i], keypoint_conf,
            previous_plate, K, object_points,
            frame_count, fps, next_plate_id
        )

        if p is not None:
            current_plates.append(p)

            if previous_plate is not None:
                matched_previous.add(p.id)

    return current_plates, plate_boxes, next_plate_id


def remove_expired_tracks(current_robots, frame_count):
    expired_plates = [
        id for id, p in plates.items()
        if frame_count - p.last_detection_frame > max_missing_frames
    ]

    for id in expired_plates:
        del plates[id]

    for bot in current_robots:
        expired_phases = [
            id for id in bot.plate_phases
            if id not in plates
        ]

        for id in expired_phases:
            del bot.plate_phases[id]


def print_robot_states(current_robots):
    for i, bot in enumerate(current_robots):
        print(
            f"Robot: {i} | "
            f"Plates: {list(bot.plates.keys())} | "
            f"Position: ({bot.x:.3f}, {bot.y:.3f}, {bot.z:.3f}) | "
            f"Velocity: ({bot.dx:.3f}, {bot.dy:.3f}, {bot.dz:.3f}) | "
            f"Yaw: {np.degrees(bot.yaw):.2f} | "
            f"Angular velocity: {bot.dyaw:.3f} | "
            f"Radius: {bot.r:.3f}"
        )


def process_grouping_result(pending):
    frame_id, completed = pending.popleft()
    matches = completed.result()

    for team, center in matches:
        print(
            f"Frame: {frame_id} | "
            f"Team: {team} | "
            f"Center: {center}"
        )


def submit_grouping(executor, pending, frame_count, frame,
                    plate_boxes, enemyColor, allyColor, maxPixDistance):

    future = executor.submit(
        grouping, frame, plate_boxes,
        enemyColor, allyColor, maxPixDistance
    )

    pending.append((frame_count, future))

    if len(pending) > 2:
        process_grouping_result(pending)


def process_frame(result, K, object_points, frame_count,
                  fps, maxPixDistance, next_plate_id):

    current_plates, plate_boxes, next_plate_id = process_plates(
        result, K, object_points, frame_count,
        fps, maxPixDistance, next_plate_id
    )

    current_robots = update_robots(
        current_plates, frame_count, 1.0 / fps
    )

    print_robot_states(current_robots)
    remove_expired_tracks(current_robots, frame_count)

    return plate_boxes, next_plate_id


def track_video(model, video_path, enemyColor, allyColor, maxPixDistance):
    video_path = Path(video_path)

    if not video_path.is_absolute():
        video_path = root / video_path

    frame_count = 0
    start_time = time.perf_counter()
    executor = ThreadPoolExecutor(max_workers=1)
    pending = deque()

    fps = get_video_fps(video_path)
    next_plate_id = 0
    object_points = get_object_points()

    plates.clear()
    robotsnplates.clear()

    K = None
    previous_resolution = None

    results = model.predict(
        source=video_path,
        imgsz=imgsz,
        conf=0.1,
        tracker=str(root / "byteTrackConfig.yaml"),
        save=True,
        device=0,
        stream=True,
        # vid_stride=skip,
    )

    try:
        for result in results:
            frame_count += 1
            frame = result.orig_img
            height, width = frame.shape[:2]

            if previous_resolution != (width, height):
                K = get_camera_matrix(width, height)
                previous_resolution = (width, height)
                plates.clear()
                robotsnplates.clear()

            plate_boxes, next_plate_id = process_frame(
                result, K, object_points, frame_count,
                fps, maxPixDistance, next_plate_id
            )

            submit_grouping(
                executor, pending, frame_count, frame,
                plate_boxes, enemyColor, allyColor, maxPixDistance
            )

        while pending:
            process_grouping_result(pending)

    finally:
        executor.shutdown(wait=True)

    elapsed = time.perf_counter() - start_time

    if elapsed > 0:
        print(
            f"Tracking speed for {video_path.name}: "
            f"{frame_count / elapsed:.2f} FPS "
            f"({frame_count} frames over {elapsed:.2f}s)"
        )

    return frame_count


def color_detection(frame):
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


def grouping(image, plate_boxes, enemyColor, allyColor, maxPixDistance):
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

    color_boxes = color_detection(image)

    if not color_boxes:
        return []

    color_boxes = [
        (color, box)
        for color, box in color_boxes
        if color in (enemyColor, allyColor)
    ]

    if not color_boxes:
        return []

    candidates = []

    for plate_index, plate_box in enumerate(plate_boxes):
        px1, py1, px2, py2 = plate_box
        center = (
            int((px1 + px2) / 2),
            int((py1 + py2) / 2)
        )

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
        root / "runs/detect/armor/armor_pose_EPOCH100-10/weights/best.engine"
    )
    model = YOLO(str(model_path))

    
    enemyColor = "red"
    allyColor = "blue"
    maxPixDistance = 30

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