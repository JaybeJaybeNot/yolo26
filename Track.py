#NOTES ON 


from ultralytics import YOLO
from pathlib import Path
import cv2
import time

root = Path(__file__).resolve().parent
#Basic Track testing
videos = [
    "[BLUR_TEST]RoboMasterTestVideo3.mp4",
    #"RoboMasterTestVideo1.mp4",
    "RoboMasterTestVideo2.mp4",
]

COLOR_ROI_WIDTH = 320
COLOR_ROI_HEIGHT = 320

class plate:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def move(self, dx, dy):
        self.x += dx
        self. y += dy


class robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def move(self, dx, dy):
        self.x += dx
        self. y += dy
    color: str
    plates: plate


robots = [ #list containing robots and matched colors 

]

def combinedTrack(
    model_path, video_path, roi_width=COLOR_ROI_WIDTH, roi_height=COLOR_ROI_HEIGHT
):
    if roi_width <= 0 or roi_height <= 0:
        raise ValueError("ROI width and height must be positive")

    model = YOLO(str(model_path))
    video_path = Path(video_path)
    if not video_path.is_absolute():
        video_path = root / video_path

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Could not read video properties: {video_path}")

    output_dir = root / "runs" / "predict"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_path.stem}_tracked_color.mp4"

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    frame_intervals = []
    previous_frame_time = None
    try:
        while True:
            success, original_frame = cap.read()
            if not success:
                break

            color_regions = detect_color_regions(original_frame)
            tracking_frame = keep_color_rois(
                original_frame, color_regions, roi_width, roi_height
            )
            results = model.track(
                source=tracking_frame,
                imgsz=480,
                conf=0.01,
                tracker=str(root / "byteTrackConfig.yaml"),
                persist=True,
                verbose=False,
            )
            result = results[0]

            # Run detection on the masked frame, but show annotations over the
            # original frame so the output video remains easy to interpret.
            frame = result.plot(img=original_frame.copy())
            for color, contour in color_regions:
                x, y, _, _ = cv2.boundingRect(contour)
                display_color = (0, 0, 255) if color == "red" else (255, 0, 0)
                cv2.drawContours(frame, [contour], -1, display_color, 2)
                cv2.putText(
                    frame, color, (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, display_color, 2,
                )
            #Matching colors with plates
            for team, (center_x, center_y), track_id in grouping(
                result, enemyColor="red", allyColor="blue", maxPixDistance=20
            ):
                label = f"{team} ({center_x}, {center_y})"
                if track_id is not None:
                    label = f"ID {track_id}: {label}"
                    #
                cv2.putText(
                    frame, label, (center_x, max(20, center_y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                )

            writer.write(frame)
            current_frame_time = time.perf_counter()
            if previous_frame_time is not None:
                frame_intervals.append(current_frame_time - previous_frame_time)
            previous_frame_time = current_frame_time
    finally:
        cap.release()
        writer.release()

    print(f"Tracking and color detection complete: {output_path}")
    if frame_intervals:
        average_interval = sum(frame_intervals) / len(frame_intervals)
        measured_fps = 1 / average_interval
        print(f"Average time between processed frames: {average_interval:.4f} seconds")
        print(f"Measured processing rate: {measured_fps:.2f} frames per second")
    else:
        print("Measured processing rate: unavailable (fewer than two frames processed)")


def keep_color_rois(image, color_regions, roi_width, roi_height):
    """Keep fixed-size regions around detected color markers and black out the rest."""
    height, width = image.shape[:2]
    masked = image.copy()
    masked[:] = 0

    roi_width = min(roi_width, width)
    roi_height = min(roi_height, height)
    for _, contour in color_regions:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        center_x = x + box_width // 2
        center_y = y + box_height // 2

        left = min(max(center_x - roi_width // 2, 0), width - roi_width)
        top = min(max(center_y - roi_height // 2, 0), height - roi_height)
        masked[top:top + roi_height, left:left + roi_width] = image[
            top:top + roi_height, left:left + roi_width
        ]
    return masked


# Detect red and blue regions across the full frame without using YOLO boxes.
def detect_color_regions(image):
    """Return (color, contour) pairs for sufficiently large red and blue regions."""
    if image is None or image.size == 0:
        return []

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_masks = {
        "red": cv2.bitwise_or(
            cv2.inRange(hsv, (0, 80, 50), (10, 255, 255)),
            cv2.inRange(hsv, (170, 80, 50), (180, 255, 255)),
        ),
        "blue": cv2.inRange(hsv, (90, 80, 50), (130, 255, 255)),
    }

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    minimum_area = max(100, int(image.shape[0] * image.shape[1] * 0.0001))
    regions = []
    for color, mask in color_masks.items():
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        regions.extend(
            (color, contour)
            for contour in contours
            if cv2.contourArea(contour) >= minimum_area
        )
    return regions

def grouping(frame, enemyColor, allyColor, maxPixDistance):
    """Match tracked plate detections to nearby color regions.

    ``frame`` is an Ultralytics Results object. Return (team, center, track_id)
    tuples, where team is "enemy" or "ally" and center is the plate center in
    image pixels. Each plate and color region is matched at most once.
    """

    #_______________________________________ERROR STUFF FOR DEVS (IGNORe)______________________
    enemyColor = enemyColor.lower()
    allyColor = allyColor.lower()
    available_colors = {"red", "blue"}
    if enemyColor not in available_colors or allyColor not in available_colors:
        raise ValueError("enemyColor and allyColor must be 'red' or 'blue'")
    if enemyColor == allyColor:
        raise ValueError("enemyColor and allyColor must be different")
    if maxPixDistance < 0:
        raise ValueError("maxPixDistance must be non-negative")
 #____________________________________________________________where grouping starts
    image = frame.orig_img #the original image for that .track frame
    boxes = frame.boxes #the boxes in that .track frame
    if image is None or boxes is None:
        return []

    plate_boxes = boxes.xyxy.cpu().numpy() #boxes.xyxy is tensor contraining bounding box coordinates. .cpu() moves tensor to cpu mem because numpy can't directly read GPU mem.
    if len(plate_boxes) == 0:
        return []

    regions = detect_color_regions(image)
    if not regions:
        return []

    track_ids = boxes.id
    track_ids = (
        track_ids.cpu().numpy().astype(int).tolist()
        if track_ids is not None
        else [None] * len(plate_boxes)
    )
    color_boxes = [
        (color, cv2.boundingRect(contour))
        for color, contour in regions
        if color in (enemyColor, allyColor)
    ]

    candidates = [] #This list contains all the matched plates and color (which determines whether they're enemies or allies)
    for plate_index, plate in enumerate(plate_boxes): #YOLO Box ID and box's coordinates, respectively
        px1, py1, px2, py2 = plate
        center = (int((px1 + px2) / 2), int((py1 + py2) / 2))
        for region_index, (color, (rx, ry, rw, rh)) in enumerate(color_boxes):
            dx = max(px1 - (rx + rw), rx - px2, 0)
            dy = max(py1 - (ry + rh), ry - py2, 0)
            distance = (dx * dx + dy * dy) ** 0.5
            if distance <= maxPixDistance:
                team = "enemy" if color == enemyColor else "ally"
                candidates.append(
                    (distance, plate_index, region_index, team, center)
                )



    matches = [] #This list contains candidates without repeated instances
    matched_plates = set()
    matched_regions = set()
    for _, plate_index, region_index, team, center in sorted(candidates):
        if plate_index in matched_plates or region_index in matched_regions:
            continue
        matched_plates.add(plate_index)
        matched_regions.add(region_index)
        matches.append((team, center, track_ids[plate_index]))
    return matches

#tracking armor
'''
def trackArmor(){

}
'''
#Implementation of "likely"



#implementation of "Region of confidence"

#IDEA: after grouping, colors are matched with armor. Let such colors be "Marked". Marked colors will be used to detect the broader robot as a whole


#main
def main():
    model_path = root / "runs/detect/[LARGE]Robomasters.v6i.yolo26_EPOCH100/weights/best.pt"
    for video in videos:
        video_path = root / str(video)
        combinedTrack(model_path, video_path)


if __name__ == "__main__":
    main()