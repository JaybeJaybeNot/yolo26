"""Entry point: camera -> detect -> track -> aim -> send.

The only executable in this package; everything else is a library.

    marc-cv --video clip.mp4
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MARC CV pipeline.")
    parser.add_argument("--video", help="video file to run on instead of a live camera")
    args = parser.parse_args()

    raise SystemExit("not implemented yet: " + (args.video or "live camera"))


if __name__ == "__main__":
    main()
