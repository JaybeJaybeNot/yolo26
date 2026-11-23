<video src="./LocalCVATandAutoLabelingTutorial.mp4" controls width="600">
  <a href="./LocalCVATandAutoLabelingTutorial.mp4">Open the tutorial video</a>
</video>


# CVAT YOLO-Pose Auto-Labeling

This folder contains the files used for auto-labeling:

- `func.py` — CVAT's Ultralytics YOLO auto-annotation function
- `best.pt` — trained pose-estimation model
- `keypoints.txt` — the four keypoint names, one per line

## Requirements

- Python 3.10 or newer
- A running CVAT server reachable from this machine
- An NVIDIA GPU and a CUDA-enabled PyTorch installation to use `device=str:cuda`

The CVAT server is a separate service; this folder contains the client-side
model and function, not the CVAT server or its Docker Compose files.

## Install the Python dependencies

From this folder, create and activate a virtual environment, then install the
requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Confirm that PyTorch can use your GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

It should print `True` when using `device=str:cuda`. If it prints `False`,
install the PyTorch build that matches your system from
[the PyTorch install selector](https://pytorch.org/get-started/locally/).
Alternatively, change the command's device to `cpu`.

## Start a local CVAT server

If CVAT is not already installed, clone and start it separately:

```bash
cd ~
git clone https://github.com/cvat-ai/cvat.git
cd ~/cvat
docker compose up -d
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

CVAT should then be available at `http://localhost:8080`. To resume it later:

```bash
cd ~/cvat
docker compose up -d
```

## Run auto-labeling

Activate the virtual environment and change to this folder before running the
command. Replace `USER` with your CVAT username and `TASKNUM` with
the task number. From the repository root, run `cd marc-cv-data`; the command's
file paths are relative to this folder.

```bash
cd marc-cv-data
source .venv/bin/activate
cvat-cli --server-host http://localhost:8080 --auth USER task auto-annotate TASKNUM --function-file ./func.py -p model=str:./best.pt -p device=str:cuda -p keypoint_names_path=str:./keypoints.txt --conf-threshold 0.4
```

For CPU inference, change `device=str:cuda` to `device=str:cpu`.
The CVAT task's labels and skeleton keypoint order must match the model and
`keypoints.txt`.

## CVAT maintenance

Run these commands from the CVAT installation directory (`~/cvat`):

```bash
docker compose ps                 # Check container status
docker compose logs -f cvat_server # Follow server logs
docker compose restart            # Restart CVAT
docker compose down               # Stop CVAT
docker system df                  # Check Docker storage usage
```

Do not use `docker compose down -v`; it deletes persistent volumes containing
CVAT projects, images, and annotations.

## Export
Once you exported your labeled data, you should see
```
TaskName/
    ├── images/
    │   └── train/
    │       ├── image1.jpg
    │       ├── image2.jpg
    │       ├── image3.jpg
    │       └── ...
    ├── labels/
    │   └── train/
    │       ├── image1.txt
    │       ├── image2.txt
    │       ├── image3.txt
    │       └── ...
    └── data.yaml
```

add val folders in in images and labels. They are used while training to evaluate accuracy.
```
mkdir -p TaskName/{images,labels}/val
```

Important: contents in data.yaml and val may vary based on what you are training.

