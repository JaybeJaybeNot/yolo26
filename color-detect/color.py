import cv2
from colorAPI import blurImage, maskColor, maskRed, morphology, precisionRectPosition

# BGR
yellow = [0, 255, 255]
red = [0, 0, 255]
blue = [255, 0, 0]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Blur image
    frame = blurImage(frame)

    # Convert BGR -> HSV
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # BLUE
    blueMask = maskColor(hsv_image, blue)
    blueMask = morphology(blueMask)
    blueContours, _ = cv2.findContours(blueMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    precisionRectPosition(frame, blueContours, blue, "BLUE")

    # RED
    redMask = maskRed(hsv_image)
    redMask = morphology(redMask)
    redContours, _ = cv2.findContours(redMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    precisionRectPosition(frame, redContours, red, "RED")

    # Show both masks together
    combinedMask = blueMask | redMask
    cv2.imshow("mask", combinedMask)
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()