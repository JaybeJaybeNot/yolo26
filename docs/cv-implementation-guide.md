# CV Implementation Guide

The computer vision pipeline processes camera input to detect targets, calculate firing solutions, and guide autonomous navigation. Each module handles one part of this chain. Implementation should follow the order below to keep dependencies clean.

## Module Implementation Order

### 1. Camera Capture (`camera/capture.py`)
The foundation. Reads frames from the Manifold Odin camera or a video file for testing.

**What it does:**
- Opens the camera feed at 30 FPS
- Yields raw frames as numpy arrays
- Handles disconnection gracefully

**Why it's first:** Everything else needs frames to work on.

---

### 2. Target Detection (`aim_assist/detector_classical.py`)
Finds enemies on the arena using color and contour analysis.

**Algorithm:**
- Convert frame to HSV (easier for color ranges)
- Threshold to isolate red/blue team colors
- Find contours and filter by size/shape
- Return bounding boxes of detected targets

**Output:** List of target boxes and their positions in the frame

---

### 3. Ballistics Calculation (`auto_aiming/ballistics.py`)
Converts target position to a firing angle, accounting for projectile drop and robot movement.

**Physics:**
- Extract 3D position from 2D detection (using PnP if depth available)
- Calculate lead needed based on target velocity
- Account for bullet flight time and gravity
- Output yaw and pitch angles for the gimbal

---

### 4. Auto-Fire Trigger (`auto_firing/trigger.py`)
Decides when the conditions are right to shoot.

**Logic:**
- Check if target is in frame and moving slowly
- Verify gimbal has reached commanded angle
- Confidence threshold: only fire if detection is stable
- Return boolean: fire or hold

---

### 5. Target Prioritization (`autonomous/prioritization.py`)
In autonomous mode, pick which enemy to shoot and route the robot accordingly.

**Strategy:**
- Score targets by distance, armor type, and threat level
- Plan a path that doesn't run into walls (use arena contour map)
- Switch targets if a higher-priority one appears
- Return next target and recommended movement

---

### 6. Board Communication (`communication/board_link.py`)
Send aiming and movement commands to the firmware over a socket.

**Protocol:**
- Pack yaw/pitch angles and movement vectors into a message
- Send at 100 Hz (keep pace with the 1 ms firmware tick)
- Handle timeouts and reconnection

---

## Testing Each Module

Build and test in order. Use recorded video for the first three so you don't need the camera hardware.

- `make test` runs the unit tests in each module's `tests/` folder
- `cv/color-detect/` has reference code for color thresholding if you get stuck
- The arena contour map is in `data/arena_maps/` for navigation

Start with detection. Get that rock solid, then everything else clicks into place.
