
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import mss
import numpy as np

# Screen region containing your video window
REGION = {
    "left": 300,
    "top": 200,
    "width": 800,
    "height": 600,
}

PORT = 8765
TARGET_FPS = 30


class VideoStream(BaseHTTPRequestHandler):
    def do_GET(self):
        print(f"Requested path: {self.path!r}")

        if self.path != "/video":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=frame"
        )
        self.end_headers()

        try:
            with mss.mss() as screen:
                while True:
                    start = time.perf_counter()

                    screenshot = screen.grab(REGION)
                    frame = np.asarray(screenshot)[:, :, :3]

                    success, encoded = cv2.imencode(
                        ".jpg",
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 85]
                    )

                    if not success:
                        continue

                    data = encoded.tobytes()

                    self.wfile.write(
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: "
                        + str(len(data)).encode()
                        + b"\r\n\r\n"
                        + data
                        + b"\r\n"
                    )

                    elapsed = time.perf_counter() - start
                    time.sleep(max(0, 1 / TARGET_FPS - elapsed))

        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), VideoStream)
    print(f"Video stream on port {PORT}")
    server.serve_forever()
