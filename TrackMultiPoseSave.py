from ultralytics import YOLO

from TrackMultiSave import root, track_video, videos


def main():
    model_path = (
        root / "runs/detect/armor/armor_pose_EPOCH100-3/weights/best.pt"
    )
    model = YOLO(str(model_path))

    enemyColor = "red"
    allyColor = "blue"
    maxPixDistance = 80
    save = True

    for video in videos:
        track_video(
            model,
            video,
            enemyColor,
            allyColor,
            maxPixDistance,
            save
        )


if __name__ == "__main__":
    main()
