"""Calculate where to aim given a target position and velocity.

Solves the ballistics problem: projectile drop due to gravity, lead for moving
targets, distance from camera to target. Takes 3D position and outputs gimbal angles.

**What to implement:**
- `calculate_aim(target_pos, target_vel, muzzle_vel)` returns (yaw, pitch)
- Use projectile motion equations to find flight time and drop
- Calculate lead: where the target will be when the bullet arrives
- Convert 3D world coords to gimbal angles using camera intrinsics

**Why it matters:** Accurate aiming depends on this. Small errors compound fast
in RoboMaster since targets move. Test with stationary targets first.
"""
