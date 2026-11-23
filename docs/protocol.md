# Jetson ↔ board protocol

**Status:** not written yet. This is the spec the firmware and CV sides agree on
before either writes code.

It must say:

- the bytes in order, with a header, and what each field means
- yaw and pitch: units, sign convention, and what they are relative to
- the fire flag, and what the board does with it
- a sequence number and a checksum
- how many packets per second, and the baud rate
- what the robot does when packets stop arriving

Implemented on the CV side in `src/marc_cv/communication/board_link.py`, and on
the firmware side under `controls/src/communication/jetson/`.
