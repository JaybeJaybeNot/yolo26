import cv2
import numpy as np

THRESH = 50       # tune. How much brighter the red channel must be than blue (or blue than red) on a 0–255 scale to count as a strong "red" or "blue" pixel.
MIN_PIXELS = 15   # minimum lit pixels to trust the result

def classify_color(frame, box, pad=0.15):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1 = max(int(x1 - pad * w), 0); y1 = max(int(y1 - pad * h), 0)
    x2 = min(int(x2 + pad * w), frame.shape[1]); y2 = min(int(y2 + pad * h), frame.shape[0])

    b, _, r = cv2.split(frame[y1:y2, x1:x2])
    red = np.count_nonzero(cv2.subtract(r, b) > THRESH)   # uint8 subtract saturates at 0
    blue = np.count_nonzero(cv2.subtract(b, r) > THRESH)

    if max(red, blue) < MIN_PIXELS:
        return None            # unknown, don't update the track
    return "red" if red > blue else "blue"