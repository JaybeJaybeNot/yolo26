"""
armor_pnp.py - batch PnP for armor plates.

Usage:
    import armor_pnp as ap
    ap.configure(width=1280, height=720, hfov=70)     # or ap.configure(K=K, dist=dist)

    boxes, kpts = detect(model, frame)                # boxes [N,4] xyxy, kpts [N,4,2]
    out = ap.PnP(boxes, kpts)                         # all plates, one call
    out.valid, out.t, out.R, out.err_px               # rows align with the inputs

Keypoint order must be: top-left, bottom-left, bottom-right, top-right.
Camera frame is OpenCV convention (x right, y down, z forward).
"""
from dataclasses import dataclass
import numpy as np
import cv2

# ----------------------------------------------------------------------------
# Config (override with configure())
# ----------------------------------------------------------------------------
SMALL_W, LARGE_W, ARMOR_H = 0.135, 0.230, 0.055   # metres; match what your keypoints mark
MIN_KPT_CONF = 0.5          # was 0.75; log your rejection rate and tune
ERR_FLOOR_PX = 1.5          # absolute floor for reprojection error
ERR_FRAC_OF_H = 0.10        # threshold also scales with plate pixel height
AMBIG_PX = 0.2              # IPPE candidates within this error are "ambiguous"
PLATE_PITCH = np.radians(15)  # plate tilt from vertical; CHECK SIGN against your mount
BOX_PAD = 0.25              # keypoints must lie within box padded by this fraction
LARGE_ASPECT = 3.3          # fallback small/large guess from keypoint aspect ratio

_K = None
_DIST = np.zeros(5)


def get_camera_matrix(width, height, hfov=70):
    f = width / (2 * np.tan(np.radians(hfov / 2)))
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], np.float64)


def configure(K=None, dist=None, width=None, height=None, hfov=70):
    """Set intrinsics once. Pass a calibrated K/dist, or width/height/hfov for a rough guess."""
    global _K, _DIST
    if K is None:
        if width is None or height is None:
            raise ValueError("give K or width+height")
        K = get_camera_matrix(width, height, hfov)
    _K = np.asarray(K, np.float64)
    _DIST = np.zeros(5) if dist is None else np.asarray(dist, np.float64)


# ----------------------------------------------------------------------------
# Output container (arrays always length N, row i belongs to input plate i)
# ----------------------------------------------------------------------------
@dataclass
class PoseResult:
    valid: np.ndarray    # [N] bool
    R: np.ndarray        # [N,3,3]  plate->camera rotation (NaN if invalid or mode="position")
    t: np.ndarray        # [N,3]    plate centre in camera frame, metres (NaN if invalid)
    err_px: np.ndarray   # [N]      RMS reprojection error (NaN in position mode)
    reason: list         # [N] str, "" if ok


# ----------------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------------
def _object_points(is_large):
    w = (LARGE_W if is_large else SMALL_W) / 2
    h = ARMOR_H / 2
    return np.array([[-w, -h, 0], [-w, h, 0], [w, h, 0], [w, -h, 0]], np.float64)


_OBJ = {False: _object_points(False), True: _object_points(True)}


def _plate_height_px(p):
    return 0.5 * (np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[3]))


def _guess_large(p):
    w = 0.5 * (np.linalg.norm(p[3] - p[0]) + np.linalg.norm(p[2] - p[1]))
    return (w / max(_plate_height_px(p), 1e-6)) > LARGE_ASPECT


def _kpts_in_box(box, p):
    x1, y1, x2, y2 = box
    pw, ph = BOX_PAD * (x2 - x1), BOX_PAD * (y2 - y1)
    return bool(np.all((p[:, 0] >= x1 - pw) & (p[:, 0] <= x2 + pw) &
                       (p[:, 1] >= y1 - ph) & (p[:, 1] <= y2 + ph)))


def _prior_cost(rvec, R_c2w):
    """How far the candidate's outward normal is from the expected world-frame tilt."""
    R, _ = cv2.Rodrigues(rvec)
    n_world = R_c2w @ (-R[:, 2])      # plate +z points into the plate, so negate
    return abs(n_world[2] - np.sin(PLATE_PITCH))


def _position_only(p, is_large):
    """Similar triangles on light-bar height. Fast, no orientation."""
    h_px = _plate_height_px(p)
    z = _K[0, 0] * ARMOR_H / h_px
    cx, cy = p.mean(axis=0)
    return np.array([(cx - _K[0, 2]) * z / _K[0, 0],
                     (cy - _K[1, 2]) * z / _K[1, 1], z])


def _solve_one(p, is_large, R_c2w):
    obj = _OBJ[bool(is_large)]
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
        obj, p, _K, _DIST, flags=cv2.SOLVEPNP_IPPE)
    if n == 0:
        return None, "ippe_failed"

    errs = np.ravel(errs)
    best = int(np.argmin(errs))
    if R_c2w is not None and len(errs) > 1:
        near = [i for i in range(len(errs)) if errs[i] <= errs[best] + AMBIG_PX]
        if len(near) > 1:
            best = min(near, key=lambda i: _prior_cost(rvecs[i], R_c2w))

    # refine only the chosen candidate
    rvec, tvec = cv2.solvePnPRefineLM(
        obj, p, _K, _DIST, rvecs[best].copy(), tvecs[best].copy())
    if not (np.all(np.isfinite(rvec)) and np.all(np.isfinite(tvec))):
        return None, "nonfinite"
    if tvec[2, 0] <= 0:
        return None, "behind_camera"

    proj, _ = cv2.projectPoints(obj, rvec, tvec, _K, _DIST)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - p) ** 2, axis=1))))
    if err > max(ERR_FLOOR_PX, ERR_FRAC_OF_H * _plate_height_px(p)):
        return None, "reproj_err"

    R, _ = cv2.Rodrigues(rvec)
    return (R, tvec.ravel(), err), ""


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def PnP(boxes, kpts, *, kpt_conf=None, is_large=None, R_c2w=None, mode="pose"):
    """
    boxes:    [N,4] xyxy
    kpts:     [N,4,2] pixels, order TL, BL, BR, TR
    kpt_conf: optional [N,4]
    is_large: optional [N] bool (e.g. from your class/number). If None, guessed from
              keypoint aspect ratio, which is unreliable at oblique angles.
    R_c2w:    optional 3x3 camera->world rotation (z up) from gimbal/IMU. Used to break
              IPPE's two-solution ambiguity with a plate-tilt prior.
    mode:     "pose" (IPPE + LM refine) or "position" (similar triangles, much faster)
    """
    if _K is None:
        raise RuntimeError("call configure() first")
    if mode not in ("pose", "position"):
        raise ValueError("mode must be 'pose' or 'position'")

    boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
    kpts = np.asarray(kpts, np.float64).reshape(-1, 4, 2)
    n = len(kpts)
    R_out = np.full((n, 3, 3), np.nan)
    t_out = np.full((n, 3), np.nan)
    err_out = np.full(n, np.nan)
    valid = np.zeros(n, bool)
    reason = [""] * n
    R_c2w = None if R_c2w is None else np.asarray(R_c2w, np.float64)

    for i in range(n):
        p = np.ascontiguousarray(kpts[i])

        if not np.all(np.isfinite(p)):
            reason[i] = "nonfinite_kpts"; continue
        if kpt_conf is not None and np.min(kpt_conf[i]) < MIN_KPT_CONF:
            reason[i] = "low_conf"; continue
        if not _kpts_in_box(boxes[i], p):
            reason[i] = "kpts_outside_box"; continue
        if not cv2.isContourConvex(p.astype(np.float32)):
            reason[i] = "not_convex"; continue
        if _plate_height_px(p) < 2:
            reason[i] = "too_small"; continue

        large = bool(is_large[i]) if is_large is not None else _guess_large(p)

        if mode == "position":
            t_out[i] = _position_only(p, large)
            valid[i] = True
            continue

        res, why = _solve_one(p, large, R_c2w)
        if res is None:
            reason[i] = why; continue
        R_out[i], t_out[i], err_out[i] = res
        valid[i] = True

    return PoseResult(valid, R_out, t_out, err_out, reason)
