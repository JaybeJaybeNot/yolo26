import cv2
from ultralytics import YOLO
from pathlib import Path
import time

root = Path(__file__).resolve().parent
# 1. Load your YOLO model
model_path = (
        root / "runs/detect/[LARGE]Robomasters.v6i.yolo26_EPOCH100/weights/best.engine"
    )
model = YOLO(str(model_path))

# 2. Capture video from webcam (0) or video file/stream URL
cap = cv2.VideoCapture(0)

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    break

  # 3. Run inference on the current frame
  results = model(frame)  # or model.predict(frame, stream=False)

  # 4. Visualize or extract data from results
  annotated_frame = results[0].plot()  # draws bounding boxes on the frame

  # 5. Display the frame in real time
  cv2.imshow("YOLO Real-Time Detection", annotated_frame)

  # Press 'q' to exit the loop
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()