"""Open a camera or a video file and yield frames.

This module handles all camera I/O. It abstracts away whether you're reading
from the live Manifold Odin camera or a recorded video file for testing.

**What to implement:**
- `Camera` class that opens the Odin camera at 30 FPS
- `VideoFile` class that reads from a .mp4 or .avi file
- A `get_frames()` generator that yields (frame, timestamp) tuples
- Handle camera disconnection and log errors without crashing

**Why it matters:** Every other module depends on frames. Get this solid first.
"""
