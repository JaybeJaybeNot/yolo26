# **RoboMaster YOLO-Pose — CVAT Auto-Labeling Commands**

This documents the setup commands we discussed and the commands needed to restart your CVAT auto-labeling agent.

## **1. Initial setup (one time only)**

Install dependencies:

```
pip install -U cvat-cli ultralytics

git clone https://github.com/cvat-ai/cvat-models.git
cd cvat-models

pip install -r detector/yolo/requirements.txt
```

Configure your CVAT connection:

```
export CVAT_SERVER_URL="https://app.cvat.ai"
export CVAT_ACCESS_TOKEN="jk7GYl5G.I66awt7cNDK7IoRotO1kiOUaEcIPwpAx"
export CVAT_ORG=""
```

Create the keypoint name file, matching your CVAT skeleton order:

```
printf "1\n2\n3\n4\n" > keypoints.txt
```

Your four keypoints are:

| Index | CVAT name | Location                  |
| ----- | --------- | ------------------------- |
| 0     | 1         | Top-left LED endpoint     |
| 1     | 2         | Bottom-left LED endpoint  |
| 2     | 3         | Bottom-right LED endpoint |
| 3     | 4         | Top-right LED endpoint    |

## **2. Register the model (one time only)**

From the `cvat-models` directory:

```
cvat-cli \
  --server-host "$CVAT_SERVER_URL" \
  --org "$CVAT_ORG" \
  function create-native "RoboMaster LED Pose" \
  --function-file detector/yolo/func.py \
  -p model=str:/home/yayugang/yolo26/runs/detect/armor/armor_pose_EPOCH100/weights/best.engine \
  -p device=str:cuda \
  -p keypoint_names_path=str:keypoints.txt
```

Replace the model path with the actual location of your checkpoint.

**Record the function ID printed by CVAT.** Do not register the same model again every time you restart your computer.

## **3. Resume auto-labeling (every session)**

Main command sequence

```

# Activate your existing Python environment
source /path/to/venv/bin/activate

# Navigate to the cloned repository
cd /path/to/cvat-models

# Configure CVAT
export CVAT_SERVER_URL="https://app.cvat.ai"
export CVAT_ACCESS_TOKEN="YOUR_CVAT_TOKEN"
export CVAT_ORG=""

# Resume the agent
# Replace FUNCTION_ID with the registered function ID
cvat-cli \
  --server-host "$CVAT_SERVER_URL" \
  --org "$CVAT_ORG" \
  function run-agent FUNCTION_ID \
  --function-file detector/yolo/func.py \
  -p model=str:/home/yayugang/yolo26/runs/detect/armor/armor_pose_EPOCH100/weights/best.engine \
  -p device=str:cuda \
  -p keypoint_names_path=str:keypoints.txt

```

The Python environment and model path are placeholders; substitute the ones used in your successful registration.

When you see:

```
INFO: Trying to acquire an annotation request of category 'batch'...
```

The agent is waiting for an annotation request. Keep the terminal open, then initiate automatic annotation from CVAT.

## **4. Useful commands**

| Action                 | Command                                                      |
| ---------------------- | ------------------------------------------------------------ |
| Check CLI installation | `cvat-cli --version`                                         |
| View function commands | `cvat-cli function --help`                                   |
| See agent options      | `cvat-cli function run-agent --help`                         |
| Check CUDA             | `python -c "import torch; print(torch.cuda.is_available())"` |
| Stop agent             | `Ctrl+C`                                                     |
| Restart agent          | Rerun the Section 3 command                                  |
| Update dependencies    | `pip install -U cvat-cli ultralytics`                        |

## **5. Configuration to preserve**

Your YOLO-Pose `data.yaml`:

```
path: /home/yayugang/yolo26/armor/armor_pose

train: images/train
val: images/val

kpt_shape: [4, 3]
flip_idx: [3, 2, 1, 0]

names:
  0: armor
```

Your CVAT skeleton is named `LEDKeys`, containing four point sublabels named `1`, `2`, `3`, and `4`.

**Important:** The commands above reproduce the setup we discussed, not a verified transcript of every command executed on your computer. Also, keep your access token private and avoid committing it to Git.

Once the model is registered successfully, **you only need Section 3 for future sessions**.