from pathlib import Path
from ultralytics import YOLO

root = Path(__file__).resolve().parent
model_path = (
    root / "runs/detect/armor/armor_pose_EPOCH100-10/weights/best.engine"
)
model = YOLO(str(model_path))