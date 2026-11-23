import cv2
import numpy as np
from util import get_limits

precisionError = 10000
blurLevel = 21

def blurImage(frame):
    return cv2.GaussianBlur(frame, (blurLevel, blurLevel), 0)

def maskColor(hsv_image, color):
    lowerLimit, upperLimit = get_limits(color=color)
    return cv2.inRange(hsv_image, lowerLimit, upperLimit)

def maskRed(hsv_image):
    # Red wraps around the HSV range, so two masks are required
    lowerRed1 = np.array([0, 120, 100]) #modify middle value for error 
    upperRed1 = np.array([10, 255, 255])
    lowerRed2 = np.array([170, 120, 100])
    upperRed2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv_image, lowerRed1, upperRed1)
    mask2 = cv2.inRange(hsv_image, lowerRed2, upperRed2)
    return mask1 | mask2

def morphology(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def precisionRectPosition(frame, colorContours, color, colorName):
    for contour in colorContours:
        # Ignore tiny regions/noise
        if cv2.contourArea(contour) > precisionError:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2

            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Draw coordinates
            cv2.putText(
                frame,
                f"{colorName} ({center_x}, {center_y})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )