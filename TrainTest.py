from pathlib import Path
import albumentations  as A
from ultralytics import YOLO
#source .venv/bin/activate
data_path = "armor/armor_pose" #The folder of data
epochs = 100
run_name = f"{data_path}_EPOCH{epochs}"

imgsize = 640

yoloMod = "yolo26n-pose.pt"

augmentations = [
    # Simulate reduced optical resolution at distance
    A.Downscale(
        scale_range=(0.45, 0.85),
        p=0.35,
    ),

    # Moderate motion blur and optical blur
    A.OneOf([
        A.MotionBlur(
            blur_limit=(3, 7),
            p=0.5,
        ),
        A.GaussianBlur(
            blur_limit=(3, 5),
            p=0.3,
        ),
        A.Defocus(
            radius=(1, 3),
            alias_blur=(0.1, 0.3),
            p=0.2,
        ),
    ], p=0.25),

    A.ImageCompression(
        quality_range=(50, 95),
        p=0.30,
    ),

    A.GaussNoise(
        std_range=(0.01, 0.04),
        p=0.20,
    ),

    # Lighting variation
    A.RandomBrightnessContrast(
        brightness_limit=0.30,
        contrast_limit=0.30,
        p=0.60,
    ),

    A.RandomGamma(
        gamma_limit=(70, 140),
        p=0.30,
    ),

    A.CLAHE(
        clip_limit=3.0,
        p=0.15,
    ),
]
def main():
    root = Path(__file__).resolve().parent
    model = YOLO(str(root / str(yoloMod)))

    model.train(
    data=str(root / str(data_path) / "data.yaml"),
    epochs=epochs,
    imgsz=640,
    device=0,
    batch=4,
    amp=True,
    cache=False,
    workers=2,
    deterministic=False,

    augmentations=augmentations,

    scale=(0.25, 1.15),
    mosaic=0.7,
    close_mosaic=15,
    translate=0.10,

    degrees=10.0,
    perspective=0.0005,
    shear=3.0,

    mixup=0.0,
    fliplr=0.0,

    box=8.0,
    cls=1.0,
    pose=40.0,
    rle = 2.0,
    kobj=2.0,

    patience=50,
    cos_lr=True,

    project=str(root / "runs" / "detect"),
    name=run_name,
    )

    print("Training complete!")

    best_path = Path(model.trainer.best)
    best = YOLO(str(best_path))
    
    #metrics = best.val(data=str(root / "robomasters.v1i.yolo26" / "data.yaml"))
    #print("mAP50:", metrics.box.map50)
    #print("mAP50-95:", metrics.box.map)

    #BENCHMARK TESTING
    '''
    videos = [
        "[BLUR_TEST]RoboMasterTestVideo3.mp4",
        "RoboMasterTestVideo1.mp4",
        "RoboMasterTestVideo2.mp4",
    ]
    for video in videos:
        print(f"Tracking in {video} with ByteTrack...")
        results = best.predict(
            source=str(root / video),
            imgsz=imgsize,
            conf=0.1,
            #tracker=str(root / "byteTrackConfig.yaml"),
            save=True,
            project=str(root / "runs" / "predict"),
            name=run_name,
            exist_ok=True,
        )
        print(results[0].speed)
    '''
    print("Testing complete!")


if __name__ == "__main__":
    main()
