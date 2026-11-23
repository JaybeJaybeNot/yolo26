import torch
import numpy as np
from ultralytics import YOLO

N_KPTS = 4
_EMPTY = (np.empty((0, 4), np.float32), np.empty((0, N_KPTS, 2), np.float32))

#attempt: packing both bounding boxes and flattened keypoints into a single tensor before calling .cpu().numpy(), paired with pre-allocated empty arrays (_EMPTY) for zero-detection frames.
def detect(model, frame): #YOLO(str(model_path)), frame
    """Returns (boxes[N,4] xyxy, kpts[N,K,2]); row i of each belongs together."""
    r = model.predict(frame, verbose=False, half=True, imgsz=640, conf=0.25)[0]
    if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
        return _EMPTY

    packed = torch.cat([r.boxes.xyxy, r.keypoints.xy.flatten(1)], dim=1).cpu().numpy()
    return packed[:, :4], packed[:, 4:].reshape(-1, N_KPTS, 2)
'''
kpts[0]            # All 4 keypoints for detection 0 -> shape (4, 2)
kpts[0, 2]         # Keypoint 2's [x, y] for detection 0 -> shape (2,)
kpts[0, 2, 0]      # Just the x coordinate of keypoint 2 for detection 0 -> scalar float32
kpts[0, 2, 1]      # Just the y coordinate of keypoint 2 for detection 0 -> scalar float32'''

#could also use a normalization algorithm that realigns the detected keypoints into a actual rectangle with rigid angle
