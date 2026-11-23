import os
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent

# Configuration
USERNAME = "yayugang"
MODEL = "runs/detect/armor/armor_pose_EPOCH100/weights/best.engine"
LINK = "http://localhost:8080/tasks/5"

DEVICE = "cuda"
CONFIDENCE = 0.4


def auto_annotate(username, model, link):
    match = re.search(r"(https?://[^/]+)/tasks/(\d+)", link)

    if not match:
        raise ValueError("Invalid CVAT task URL")

    server, task_id = match.groups()

    cvat_models = root / "cvat-models"
    function_file = cvat_models / "detector/yolo/func.py"
    keypoints_file = cvat_models / "keypoints.txt"
    model_path = root / model

    # Generate keypoint names: TL, BL, BR, TR
    keypoints_file.write_text("1\n2\n3\n4\n")

    # Verify required files
    for path in [function_file, model_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    # Remove previous CVAT Online token
    env = os.environ.copy()
    env.pop("CVAT_ACCESS_TOKEN", None)

    command = [
        "cvat-cli",
        "--server-host", server,
        "--auth", username,
        "task", "auto-annotate", task_id,
        "--function-file", str(function_file),
        "-p", f"model=str:{model_path}",
        "-p", f"device=str:{DEVICE}",
        "-p", f"keypoint_names_path=str:{keypoints_file}",
        "--conf-threshold", str(CONFIDENCE)
    ]

    subprocess.run(
        command,
        cwd=cvat_models,
        env=env,
        check=True
    )


if __name__ == "__main__":
    auto_annotate(USERNAME, MODEL, LINK)