"""Send aiming and movement commands to the firmware.

Everything the CV pipeline calculates is useless until it reaches the robot's
firmware. This module packs up gimbal angles, fire commands, and movement vectors,
wraps them in a frame, and sends them over the socket.

**What to implement:**
- `BoardLink` class that opens a socket to the board (localhost:5000)
- `send_aim(yaw, pitch, fire)` packs and sends the aiming packet
- Protocol: 1-byte header, payload, 2-byte CRC16, 1-byte footer
- Handle timeouts gracefully: log, don't crash, wait for reconnect

**Why it matters:** This is where theory meets hardware. The firmware waits for
these packets. Send them reliably and on time (at least 10 Hz).
"""
