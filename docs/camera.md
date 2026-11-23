# Camera

**Status:** not written yet.

It must say:

- which camera and lens we use, and where it is mounted
- the intrinsic matrix and distortion coefficients, and the YAML file holding them
- field of view in degrees
- exposure and gain that stop a spinning plate from blurring
- resolution and frame rate we run at, and the latency that costs

Calibration code lives in `src/marc_cv/camera/calibration.py`.
