"""Detect enemy robots using color thresholding and contour analysis.

Classical computer vision approach: convert to HSV, threshold by team color,
find contours, filter by size and shape. Fast, reliable, no GPU needed.

**What to implement:**
- `detect_targets(frame, team_color)` function that returns list of bounding boxes
- HSV color range for red and blue teams (constants at top)
- Contour filtering: remove noise, keep only armor-sized shapes
- Return format: [(x, y, width, height), ...] in pixel coordinates

**Why it matters:** This runs at 30 FPS. It has to be fast and work in variable
lighting. Once you have boxes, everything else (aiming, firing) can work.
"""
