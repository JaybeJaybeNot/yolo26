"""Decide when to fire based on target stability and gimbal alignment.

Don't just shoot whenever you see something. Wait until the target is stable
(not flickering between detections), the gimbal has settled on the angle,
and confidence is high enough to justify expending ammo.

**What to implement:**
- `should_fire(detection_stable, aim_error, confidence)` returns boolean
- Track detection history: require 3+ consecutive frames seeing the same target
- Gimbal alignment threshold: fire only if aim error is below 2 degrees
- Confidence minimum: detection certainty above 80%

**Why it matters:** Wasting shots costs you rounds and credibility. Lock in,
be sure, then fire. This is the gate between seeing and shooting.
"""
