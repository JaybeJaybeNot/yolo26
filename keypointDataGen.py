
import cv2
from pathlib import Path
from ultralytics import YOLO

model = YOLO(
    "runs/detect/armor/armor_pose_EPOCH100/weights/best.engine"
)

video = cv2.VideoCapture("RoboMasterTestVideo2.mp4")

output = Path("auto_dataset")
image_dir = output / "images"
label_dir = output / "labels"

image_dir.mkdir(parents=True, exist_ok=True)
label_dir.mkdir(parents=True, exist_ok=True)

frame_id = 0
sample_interval = 10

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame_id += 1

    if frame_id % sample_interval != 0:
        continue

    result = model.predict(
        frame,
        imgsz=640,
        conf=0.4,
        verbose=False
    )[0]

    h, w = frame.shape[:2]
    labels = []

    if result.keypoints is None:
        continue

    for box, keypoints in zip(
        result.boxes,
        result.keypoints
    ):
        if len(keypoints.xy[0]) != 4:
            continue

        cls = int(box.cls.item())

        # Normalized bounding box
        xc, yc, bw, bh = box.xywhn[0].tolist()

        # Normalized keypoints
        points = keypoints.xyn[0].tolist()

        # Visibility information if available
        if keypoints.has_visible:
            visibility = keypoints.data[0, :, 2].tolist()
        else:
            visibility = [2] * 4

        values = [cls, xc, yc, bw, bh]

        for (x, y), v in zip(points, visibility):
            values.extend([x, y, int(v > 0) * 2])

        labels.append(
            " ".join(map(str, values))
        )

    if not labels:
        continue

    filename = f"frame_{frame_id:06d}"

    cv2.imwrite(
        str(image_dir / f"{filename}.jpg"),
        frame
    )

    (label_dir / f"{filename}.txt").write_text(
        "\n".join(labels) + "\n"
    )

video.release()
