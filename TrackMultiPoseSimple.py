from pathlib import Path
from ultralytics import YOLO
import numpy as np
import cv2

imgsz = 544
root = Path(__file__).resolve().parent

videos = [
    "1.mp4",
    "dji_best_local.mp4",
    "[BLUR_TEST]RoboMasterTestVideo3.mp4",
    "RoboMasterTestVideo2.mp4",
]

MIN_KEYPOINT_CONF = 0.75
MAX_REPROJECTION_ERROR = 1.5
POSE_AMBIGUITY_THRESHOLD = 0.2
INITIAL_RADIUS = 0.23

CAMERA_TO_ROBOT = np.array([
    [0, 0, 1],
    [-1, 0, 0],
    [0, -1, 0]
], dtype=float)

SMALL_ARMOR_WIDTH = 0.135
LARGE_ARMOR_WIDTH = 0.230
ARMOR_HEIGHT = 0.055

dist_coeffs = np.zeros(5, dtype=np.float64)


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

    def updatePose(self, R, t):
        self.R = R.copy()
        self.t = t.copy()

        self.relativeX, self.relativeY, self.relativeZ = map(float, t)

        # Euler angles in degrees
        angles = cv2.RQDecomp3x3(R)[0]
        self.relativerX, self.relativerY, self.relativerZ = map(float, angles)

        self.position = CAMERA_TO_ROBOT @ t
        normal = CAMERA_TO_ROBOT @ R[:, 2]
        self.yaw = float(np.arctan2(normal[1], normal[0]))


class robot:
    def __init__(self, x, y, z, dx, dy, dz, yaw, dyaw, r, color):
        self.x = float(x)
        self.y = float(y)
        self.dx = float(dx)
        self.dy = float(dy)
        self.z = float(z)
        self.dz = float(dz)
        self.yaw = float(yaw)
        self.dyaw = float(dyaw)
        self.r = float(r)

        self.color = color
        self.plates = {}
        self.plate_phases = {}
        self.last_detection_frame = 0

        self.x_prev = np.array(self.stateVector(), dtype=float)
        self.x_pred = self.x_prev.copy()
        self.P = np.eye(9)

        self.process_noise = [1.0, 1.0, 0.5, 1.0]
        self.radius_noise = 1e-5
        self.measurement_std = [0.03, 0.03, 0.03, 0.05]

    def stateVector(self):
        return [self.x, self.dx, self.y, self.dy, self.z, self.dz,
                self.yaw, self.dyaw, self.r]

    def setState(self, x):
        self.x, self.dx, self.y, self.dy = map(float, x[:4])
        self.z, self.dz, self.yaw, self.dyaw = map(float, x[4:8])
        self.r = float(np.clip(x[8], 0.12, 0.35))
        self.x_prev = np.array(self.stateVector(), dtype=float)


plates = []
PnPCalcs = []


def get_camera_matrix(width, height, hfov=70):
    f = width / (2 * np.tan(np.radians(hfov / 2)))
    return np.array([
        [f, 0, width / 2],
        [0, f, height / 2],
        [0, 0, 1]
    ], dtype=np.float64)


def get_object_points(is_large=False):
    w = LARGE_ARMOR_WIDTH if is_large else SMALL_ARMOR_WIDTH
    h = ARMOR_HEIGHT

    return np.array([
        [-w / 2, -h / 2, 0],  # Top left
        [-w / 2,  h / 2, 0],  # Bottom left
        [ w / 2,  h / 2, 0],  # Bottom right
        [ w / 2, -h / 2, 0]   # Top right
    ], dtype=np.float64)


def reprojection_error(object_points, image_points, K, dist_coeffs, rvec, tvec):
    projected_points, _ = cv2.projectPoints(
        object_points, rvec, tvec, K, dist_coeffs
    )
    difference = projected_points.reshape(-1, 2) - image_points
    return float(np.sqrt(np.mean(np.sum(difference ** 2, axis=1))))


def pose_difference(R, t, previous_pose):
    previous_R, previous_t = previous_pose
    rotation_difference = previous_R.T @ R

    angle = np.arccos(
        np.clip((np.trace(rotation_difference) - 1) / 2, -1.0, 1.0)
    )
    translation_difference = (
        np.linalg.norm(t - previous_t)
        / max(np.linalg.norm(previous_t), 1e-6)
    )

    return angle + translation_difference


def refine_pose_lm(object_points, image_points, K, dist_coeffs, rvec, tvec,
                   max_iters=20, lambda_init=1e-3, tol=1e-10):
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
        residual = (proj.reshape(-1, 2) - img).ravel()
        return residual, jac[:, :6]

    r, J = evaluate(x)
    cost = r @ r
    lam = lambda_init

    for _ in range(max_iters):
        converged = False
        A = J.T @ J
        g = J.T @ r

        improved = False
        for _ in range(10):
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
                converged = (cost - cost_new) < tol * max(cost, 1e-12)
                x, r, J, cost = x_new, r_new, J_new, cost_new
                lam = max(lam / 10.0, 1e-12)
                improved = True
                break

            lam *= 10.0

        if not improved or converged:
            break

    return x[:3].reshape(3, 1), x[3:].reshape(3, 1)


def PnP(image_points, K, dist_coeffs, object_points,
        keypoint_conf=None, previous_pose=None):
    image_points = np.array(image_points, dtype=np.float64).reshape(4, 2)
    object_points = np.array(object_points, dtype=np.float64).reshape(4, 3)

    if not np.all(np.isfinite(image_points)):
        return None

    if keypoint_conf is not None:
        if not np.all(np.isfinite(keypoint_conf)):
            return None
        if np.any(keypoint_conf < MIN_KEYPOINT_CONF):
            return None

    if not cv2.isContourConvex(image_points.astype(np.float32)):
        return None

    # Closed-form initial poses from IPPE (up to 2 candidates)
    success, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        object_points, image_points, K, dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE
    )

    if not success:
        return None

    candidates = []

    for rvec, tvec in zip(rvecs, tvecs):
        if not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
            continue

        rvec, tvec = refine_pose_lm(
            object_points, image_points, K, dist_coeffs, rvec, tvec
        )

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.flatten()

        if not np.all(np.isfinite(R)) or not np.all(np.isfinite(t)):
            continue

        camera_points = (R @ object_points.T).T + t
        if np.any(camera_points[:, 2] <= 0):
            continue

        error = reprojection_error(
            object_points, image_points, K, dist_coeffs, rvec, tvec
        )

        if not np.isfinite(error) or error > MAX_REPROJECTION_ERROR:
            continue

        candidates.append((error, R, t))

    if not candidates:
        return None

    candidates.sort(key=lambda candidate: candidate[0])
    best_error, best_R, best_t = candidates[0]

    # Resolve ambiguity using the previous pose, if one is given
    if previous_pose is not None:
        ambiguous_candidates = [
            c for c in candidates
            if c[0] <= best_error + POSE_AMBIGUITY_THRESHOLD
        ]

        _, best_R, best_t = min(
            ambiguous_candidates,
            key=lambda c: pose_difference(c[1], c[2], previous_pose)
        )

    return best_R, best_t


def track_video(model, video_path):
    video_path = Path(video_path)
    if not video_path.is_absolute():
        video_path = root / video_path

    object_points = get_object_points()
    K = None

    results = model.predict(
        source=video_path,
        imgsz=imgsz,
        conf=0.1,
        #tracker=str(root / "byteTrackConfig.yaml"),
        save=False,
        device=0,
        stream=True,
        # vid_stride=skip,
    )

    for frame_count, result in enumerate(results, start=1):
        if K is None:
            height, width = result.orig_img.shape[:2]
            K = get_camera_matrix(width, height)

        if result.keypoints is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        points = result.keypoints.xy.cpu().numpy()
        confs = (
            result.keypoints.conf.cpu().numpy()
            if result.keypoints.conf is not None
            else [None] * len(points)
        )

        for box, pts, conf in zip(boxes, points, confs):
            pose = PnP(pts, K, dist_coeffs, object_points, conf)
            if pose is None:
                continue

            R, t = pose
            p = plate(pts, len(plates), frame_count,
                      box[2] - box[0], box[3] - box[1])
            p.updatePose(R, t)

            plates.append(p)
            PnPCalcs.append((R, t))
            print(f"Frame {frame_count} | Position: {t}")


def main():
    model_path = (
        root / "runs/detect/armor/armor_pose_EPOCH100-10/weights/best.engine"
    )
    model = YOLO(str(model_path))

    for video in videos:
        track_video(model, video)


if __name__ == "__main__":
    main()