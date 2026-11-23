import cv2

def detect_colors(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red wraps around the HSV hue boundary
    red = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 80, 50), (10, 255, 255)),
        cv2.inRange(hsv, (170, 80, 50), (179, 255, 255))
    )

    blue = cv2.inRange(hsv, (90, 80, 50), (130, 255, 255))

    return red, blue


cap = cv2.VideoCapture("RoboMasterTestVideo2.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    red, blue = detect_colors(frame)

    cv2.imshow("Red Mask", red)
    cv2.imshow("Blue Mask", blue)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()