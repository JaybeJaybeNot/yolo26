# Local CVAT Deployment

## Initial Installation

```bash
cd ~
git clone https://github.com/cvat-ai/cvat.git
cd cvat
docker compose up -d
```

## Create Administrator Account

```bash
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

Access CVAT at http://localhost:8080

## Resume CVAT

```bash
cd ~/cvat
docker compose up -d
```

## AI Autolabeling

```bash
cvat-cli --server-host http://localhost:8080 --auth USER:PASSWORD task auto-annotate TASKNUM --function-file /home/yayugang/cvat-models/detector/yolo/func.py -p model=str:/home/yayugang/yolo26/runs/detect/armor/armor_pose_EPOCH100/weights/best.pt -p device=str:cuda -p keypoint_names_path=str:/home/yayugang/yolo26/cvat-models/keypoints.txt --conf-threshold 0.4
```
where TASKNUM 6 is replaced by task number, --auth replaced by actual username and password, model by path to best.pt, --function-file replcaed by func.py, and keypoint_names_path replaced by path to keypoints.txt

## Stop CVAT

```bash
cd ~/cvat
docker compose down
```

## Restart CVAT

```bash
cd ~/cvat
docker compose restart
```

## Check Container Status

```bash
docker compose ps
```

## Monitor Server Logs

```bash
docker compose logs -f cvat_server
```

## Check Storage Usage

```bash
docker system df
```

## Warning

Do not use `docker compose down -v`, as this deletes persistent
volumes containing CVAT projects, images, and annotations.

# cv

Computer vision for the robot: armor plate detection and aiming, running on the
companion computer and sending results to `controls`.

Nothing here is implemented yet — the folders are the plan. See
[docs/overview.md](docs/overview.md) for what this area is, and
[onboarding/overview.md](onboarding/overview.md) to get started.

## Setup

From this folder:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running

```bash
marc-cv --video clip.mp4        # run the pipeline on a recorded clip
pytest                          # unit tests
```

## Layout

Folders match the CV groups in [the architecture diagram](../docs/software-architecture.md),
which are also the `cv:*` PR scopes — so the folder you touch tells you the
scope for your PR title.

| Folder | PR scope | Holds |
|---|---|---|
| `src/marc_cv/camera/` | — | capture, calibration, intrinsics |
| `src/marc_cv/aim_assist/` | `cv:aim-assist` | detection, tracking |
| `src/marc_cv/auto_aiming/` | `cv:auto-aiming` | PnP, prediction, ballistics |
| `src/marc_cv/auto_firing/` | `cv:auto-firing` | trigger logic |
| `src/marc_cv/autonomous/` | `cv:autonomous` | target prioritization |
| `src/marc_cv/communication/` | `cv:communications` | serial link to the board, game data |
| `src/marc_cv/debugging/` | `cv:communications` | overlays, benchmarks |

`src/marc_cv/run.py` is the only entry point. Everything else is a library.

Unit tests live in a `tests/` folder inside the module they test, named
`<thing>_tests.py` — same rule as `controls/`, see
[docs/rules/codebase-structure.md](../docs/rules/codebase-structure.md).

## What is not committed here

Video clips and model weights are too big for git. Footage lives in the team
Drive; weights are linked from [models/README.md](models/README.md) with their
hashes.



