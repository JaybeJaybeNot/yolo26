from ultralytics import YOLO
from pathlib import Path

root = Path(__file__).resolve().parent
data_path = "armor/armor_pose"

model = YOLO("runs/detect/armor/armor_pose_EPOCH100-10/weights/best.pt")
model.export(format="engine", half=True, imgsz = 416, data=str(root / str(data_path) / "data.yaml"), nms = False, dynamic = False, workspace=4)
#python -c "import torch, tensorrt; from modelopt.onnx.quantization import quantize; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available()); print('TensorRT:', tensorrt.__version__); print('ModelOpt: OK')"