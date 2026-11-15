"""Smoke test: webcam -> OBS Virtual Camera via pyvirtualcam.

Usage:
    python test_vcam.py            # run until Ctrl+C
    python test_vcam.py --seconds 10
    python test_vcam.py --no-webcam   # send a test pattern instead of the webcam
"""
import argparse
import time

import cv2
import numpy as np
import pyvirtualcam
from pyvirtualcam import PixelFormat

W, H, FPS = 1280, 720, 30


def test_pattern(t: float) -> np.ndarray:
    frame = np.full((H, W, 3), 245, np.uint8)
    x = int((t * 200) % W)
    cv2.circle(frame, (x, H // 2), 60, (40, 120, 230), -1)
    cv2.putText(frame, "Magic Board Cam test", (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (30, 30, 30), 3)
    return frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=0, help="stop after N seconds (0 = forever)")
    ap.add_argument("--no-webcam", action="store_true")
    args = ap.parse_args()

    cap = None
    if not args.no_webcam:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        if not cap.isOpened():
            print("Could not open webcam; falling back to test pattern.")
            cap = None

    start = time.time()
    sent = 0
    with pyvirtualcam.Camera(width=W, height=H, fps=FPS, fmt=PixelFormat.BGR) as cam:
        print(f"Sending to: {cam.device}  ({W}x{H} @ {FPS} fps). Ctrl+C to stop.")
        try:
            while True:
                t = time.time() - start
                if args.seconds and t > args.seconds:
                    break
                if cap is not None:
                    ok, frame = cap.read()
                    if not ok:
                        print("Webcam read failed.")
                        break
                    frame = cv2.resize(frame, (W, H))
                    cv2.putText(frame, "LIVE via Magic Board Cam", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 3)
                else:
                    frame = test_pattern(t)
                cam.send(frame)
                cam.sleep_until_next_frame()
                sent += 1
        except KeyboardInterrupt:
            pass
    if cap is not None:
        cap.release()
    elapsed = time.time() - start
    print(f"Sent {sent} frames in {elapsed:.1f}s ({sent / max(elapsed, 1e-6):.1f} fps).")


if __name__ == "__main__":
    main()
