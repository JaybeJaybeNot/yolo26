"""Pick which enemy to shoot and plan a path to reach them.

In autonomous mode you don't wait for manual aim commands. Scan all visible
targets, score them by distance, threat level, and armor type, pick the
highest priority, and navigate there without hitting walls.

**What to implement:**
- `rank_targets(detections, arena_map)` returns sorted list with scores
- Scoring: distance (closer = better), threat (full health = worse), type (hero > other)
- Pathfinding: use arena contour map to avoid obstacles
- Return target ID, recommended movement vector, and confidence

**Why it matters:** Autonomous mode wins tournaments. This is the brain that
decides what to do when nobody is giving commands.
"""
