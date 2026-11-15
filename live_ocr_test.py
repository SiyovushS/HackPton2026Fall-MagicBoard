"""Live webcam test for the OCR engine.

    .venv/bin/python live_ocr_test.py              # webcam 0, window only
    .venv/bin/python live_ocr_test.py --vcam       # also send the view to OBS Virtual Camera (Zoom)
    .venv/bin/python live_ocr_test.py --camera 1   # another camera

In the window:
    drag with the mouse   select the region to read
    M / T / A             read the region as Math / Text / Auto (runs in the background)
    F                     read it from 3 frames and vote (text)
    L                     artifact label check against the names in --labels
    C                     clear the selection and result
    Q or Esc              quit

Every read is saved to live_captures/ (the crop + a line in log.jsonl), so good examples
can be added to tests/ocr later.
"""
import argparse
import json
import resource
import threading
import time
from pathlib import Path

import cv2
import numpy as np

WINDOW = "Live OCR test  (drag = select, M/T/A = read, F = 3-frame vote, L = label, C = clear, Q = quit)"
DISPLAY_WIDTH = 1280
CAPTURES = Path(__file__).resolve().parent / "live_captures"


class State:
    def __init__(self):
        self.engine = None
        self.status = "Loading OCR model (first run downloads ~17 GB)..."
        self.busy = False
        self.result = None          # (label, text, seconds)
        self.drag_start = None
        self.box = None             # (x0, y0, x1, y1) in full-resolution frame coordinates
        self.lock = threading.Lock()


def load_engine(state):
    t = time.time()
    try:
        from ocr_engine import OCREngine
        state.engine = OCREngine()
        state.status = f"Model ready ({time.time() - t:.0f}s). Drag a box around writing, then press M, T or A."
    except Exception as e:  # show the problem in the window instead of dying silently
        state.status = f"Model failed to load: {type(e).__name__}: {e}"


def read_in_background(state, crops, mode, labels):
    """Run one OCR job on a worker thread so the video keeps playing."""
    def job():
        t = time.time()
        try:
            if mode == "label":
                out = state.engine.recognize_label(crops[0], labels)
                text = f"label -> {out!r}"
            elif mode == "frames":
                text = state.engine.recognize_frames(crops, hint="text") or "(nothing read)"
            else:
                hint = {"math": "math", "text": "text", "auto": None}[mode]
                text = state.engine.recognize(crops[0], hint=hint) or "(nothing read)"
                if mode == "math" and state.engine.has_answer_blank(crops[0]):
                    text += "   [answer blank found -> autocomplete]"
        except Exception as e:
            text = f"error: {type(e).__name__}: {e}"
        secs = time.time() - t
        with state.lock:
            state.result = (mode, text, secs)
            state.busy = False
        save_capture(crops[0], mode, text, secs)
        print(f"[{mode}] {secs:.1f}s  {text}", flush=True)

    state.busy = True
    threading.Thread(target=job, daemon=True).start()


def save_capture(crop, mode, text, secs):
    CAPTURES.mkdir(exist_ok=True)
    name = time.strftime("%Y%m%d-%H%M%S") + f"_{mode}.png"
    cv2.imwrite(str(CAPTURES / name), crop)
    with open(CAPTURES / "log.jsonl", "a") as f:
        f.write(json.dumps({"image": name, "mode": mode, "result": text, "seconds": round(secs, 2)}) + "\n")


def open_camera(index):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)   # ask for the highest common resolution:
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # OCR needs every pixel of the writing
    return cap


def memory_gb():
    # ru_maxrss is in bytes on macOS (kilobytes on Linux): peak memory of this process
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if peak > 1e8 else peak / 1e6


def draw(view, state, scale, fps):
    h, w = view.shape[:2]
    if state.box:
        x0, y0, x1, y1 = (int(v * scale) for v in state.box)
        cv2.rectangle(view, (x0, y0), (x1, y1), (0, 200, 255), 2)
    lines = [state.status if state.engine is None else
             ("Reading..." if state.busy else "Drag a box, then M (math) / T (text) / A (auto) / F (3-frame) / L (label)")]
    if state.result:
        mode, text, secs = state.result
        lines.append(f"[{mode}, {secs:.1f}s]  {text}")
    lines.append(f"{fps:4.1f} fps   peak memory {memory_gb():.1f} GB")
    pad = 8
    box_h = 28 * len(lines) + pad
    overlay = view.copy()
    cv2.rectangle(overlay, (0, h - box_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, view, 0.4, 0, view)
    for i, line in enumerate(lines):
        y = h - box_h + pad + 20 + 28 * i
        cv2.putText(view, line[:140], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--vcam", action="store_true", help="also send the view to OBS Virtual Camera")
    ap.add_argument("--labels", default="apple,house,cat.jpg,lec.2,maple,pineapple",
                    help="comma-separated saved artifact names for the L key")
    args = ap.parse_args()
    labels = [s.strip() for s in args.labels.split(",") if s.strip()]

    # Load the model BEFORE opening the camera: loading saturates the Mac for ~20-30 s,
    # and the camera stops delivering frames if it is starved that long.
    state = State()
    print("Loading the OCR model (first run downloads ~17 GB)...", flush=True)
    load_engine(state)
    print(state.status, flush=True)
    if state.engine is None:
        raise SystemExit(1)

    cap = open_camera(args.camera)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not read from the camera. Check System Settings > Privacy & Security > Camera.")
    fh, fw = frame.shape[:2]
    print(f"Camera {args.camera}: {fw}x{fh}", flush=True)
    scale = min(1.0, DISPLAY_WIDTH / fw)       # display scale; selections map back to full resolution

    def on_mouse(event, x, y, flags, _):
        fx, fy = int(x / scale), int(y / scale)
        if event == cv2.EVENT_LBUTTONDOWN:
            state.drag_start = (fx, fy)
        elif event == cv2.EVENT_MOUSEMOVE and state.drag_start and flags & cv2.EVENT_FLAG_LBUTTON:
            x0, y0 = state.drag_start
            state.box = (min(x0, fx), min(y0, fy), max(x0, fx), max(y0, fy))
        elif event == cv2.EVENT_LBUTTONUP and state.drag_start:
            state.drag_start = None

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)

    vcam = None
    if args.vcam:
        import pyvirtualcam
        vcam = pyvirtualcam.Camera(width=int(fw * scale), height=int(fh * scale), fps=30,
                                   fmt=pyvirtualcam.PixelFormat.BGR)
        print(f"Sending to {vcam.device}", flush=True)

    frames_for_vote = []   # recent crops of the selected region, for the 3-frame vote
    voted_box = None
    last, fps = time.time(), 0.0
    failures, reopens = 0, 0
    try:
        while True:
            ok, new_frame = cap.read()
            if not ok:
                # brief hiccups happen under heavy load (e.g. while OCR runs): wait, then reopen
                failures += 1
                if failures % 40 == 0:
                    if reopens >= 3:
                        print("Camera stopped delivering frames (reopened 3 times). Giving up.")
                        break
                    print("Camera not responding, reopening it...", flush=True)
                    cap.release()
                    cap = open_camera(args.camera)
                    reopens += 1
                time.sleep(0.05)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue
            failures, frame = 0, new_frame
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-3)
            last = now

            crop = None
            if state.box != voted_box:  # new selection: don't mix crops of different regions
                frames_for_vote, voted_box = [], state.box
            if state.box and state.box[2] - state.box[0] > 10 and state.box[3] - state.box[1] > 10:
                x0, y0, x1, y1 = state.box
                crop = frame[y0:y1, x0:x1].copy()
                frames_for_vote = (frames_for_vote + [crop])[-15:]

            view = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame.copy()
            draw(view, state, scale, fps)
            cv2.imshow(WINDOW, view)
            if vcam:
                vcam.send(view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                state.box, state.result = None, None
            ready = state.engine is not None and not state.busy and crop is not None
            if key in (ord("m"), ord("t"), ord("a"), ord("l")) and ready:
                mode = {ord("m"): "math", ord("t"): "text", ord("a"): "auto", ord("l"): "label"}[key]
                read_in_background(state, [crop], mode, labels)
            if key == ord("f") and ready:
                # three frames spread over the last ~0.5 s of video
                picks = frames_for_vote[::max(1, len(frames_for_vote) // 3)][-3:] or [crop]
                read_in_background(state, picks, "frames", labels)
    finally:
        cap.release()
        if vcam:
            vcam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
