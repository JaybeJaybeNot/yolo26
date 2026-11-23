from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from models import pidinet_converted

ROOT = Path(__file__).resolve().parent
VIDEO = Path('/home/yayugang/yolo26/[BLUR_TEST]RoboMasterTestVideo3.mp4')
WEIGHTS = ROOT / 'trained_models/table5_baseline.pth'
OUTPUT = ROOT / 'test/[BLUR_TEST]RoboMasterTestVideo3.mp4'


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Build the converted baseline PiDiNet model once.
    args = SimpleNamespace(config='baseline', sa=True, dil=True)
    model = pidinet_converted(args)
    checkpoint = torch.load(WEIGHTS, map_location='cpu', weights_only=False)

    state = {
        k.removeprefix('module.'): v
        for k, v in checkpoint['state_dict'].items()
    }

    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {VIDEO}')

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise RuntimeError('Input video does not report a valid FPS')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(OUTPUT), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f'Cannot create output: {OUTPUT}')

    try:
        with torch.inference_mode():
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                # Same RGB + ImageNet normalization as the official Custom_Loader.
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
                image = image.to(device=device, dtype=torch.float32).div_(255.0)
                image = (image - mean) / std

                edges = model(image)[-1][0, 0]
                edge_gray = (edges.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                writer.write(cv2.cvtColor(edge_gray, cv2.COLOR_GRAY2BGR))
    finally:
        cap.release()
        writer.release()

    print(f'Saved: {OUTPUT}')


if __name__ == '__main__':
    main()
