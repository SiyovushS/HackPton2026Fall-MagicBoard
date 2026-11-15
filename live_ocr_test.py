"""Live webcam test for the OCR engine.

    .venv/bin/python live_ocr_test.py              # webcam 0, window only
    .venv/bin/python live_ocr_test.py --vcam       # also send the view to OBS Virtual Camera (Zoom)
    .venv/bin/python live_ocr_test.py --camera 1   # another camera

In the window:
    drag with the mouse   select the region to read
    M / T / A             read the region as Math / Text / Auto (runs in the background)
    F                     read it from 3 frames and vote (text)
    L                     artifact label check against the names in --labels
    V                     VISION MODE (recommended): the combined pipeline (board_reader.py). Keep the
                          camera still; the board at the moment you press V is the "before" picture.
                          When the board changes and settles, Qwen3-VL finds the new items (15-40 s).
    R                     LIVE MODE (rules): read the board continuously. Each region
                          is read once it has been still for ~1.5 s and no person covers it; it is
                          read again only if its writing changes. Press R again to stop.
    D                     live preview of what the snippeter finds (boxes, graph axes, writing)
    S                     snip the whole board once and read every region
    C                     clear the selection and results
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

WINDOW = "Live OCR test  (R = live mode, S = read all once, D = show snippets, drag + M/T/A/F/L, C = clear, Q = quit)"
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
        self.regions = None         # (boxes, axes, writing) from snippet.snip_board
        self.readings = {}          # ("box", i) / ("writing", i) -> text read there
        self.preview = False        # D: keep re-detecting regions on live video
        self.latest = None          # newest camera frame, for the preview thread
        self.recent = []            # (sharpness, frame) for the last ~0.5 s: snip the sharpest
        self.live = False           # R: continuous reading with person masking + stability
        self.tracker = None
        self.persons = None         # people.PersonFinder, created on first use
        self.vision = False         # V: combined pipeline (board_reader.BoardReader)
        self.reader = None
        self.person_mask = None


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


def preview_loop(state):
    """D key: re-run the snippeter on the newest frame twice a second (no OCR, ~0.1 s each)."""
    from snippet import snip_board
    while state.preview:
        frame = state.latest
        if frame is not None and not state.busy:
            regions = snip_board(frame)
            with state.lock:
                if state.preview:
                    state.regions, state.readings = regions, {}
        time.sleep(0.5)


def sharpness(frame):
    """How sharp a frame is (variance of the Laplacian on a small grey copy). Camera motion
    blurs faint chalk away: blurry frames (17-19 vs 25 here) lost the box and split expressions."""
    small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (640, 360))
    return float(cv2.Laplacian(small, cv2.CV_64F).var())


def sharpest(state):
    """The sharpest of the recent frames (or the newest, if none recorded yet)."""
    with state.lock:
        recent = list(state.recent)
    if not recent:
        return state.latest
    return max(recent, key=lambda sf: sf[0])[1]


def person_mask(state, frame):
    if state.persons is None:
        from people import PersonFinder
        state.persons = PersonFinder()
    return state.persons.mask(frame)


def live_loop(state):
    """R key: the real pipeline. A few times a second: mask people, snip the board, update the
    stability tracker; read each region once it is ready (one at a time, newest frame)."""
    from snippet import snip_board
    from tracker import BoardTracker
    state.tracker = BoardTracker()
    print("[live] started: regions are read once still for ~1.5 s and not covered by a person", flush=True)
    while state.live:
        frame = sharpest(state)
        if frame is None:
            time.sleep(0.1)
            continue
        frame = frame.copy()
        person = person_mask(state, frame)
        snipped = snip_board(frame, ignore=person)
        with state.lock:
            state.tracker.update(snipped, frame, person)
            state.person_mask = person
            ready = state.tracker.ready()
        for r in ready[:1]:                      # one read per loop, so new frames keep coming
            t = time.time()
            if r.kind == "box":
                label = state.engine.recognize(r.item.label_crop, hint="text") or "(no label read)"
                text = f"{'DEFINE' if r.item.has_content else 'CALL'} artifact, label: {label}"
            else:
                text = state.engine.recognize(r.item.crop, hint=r.item.hint) or "(nothing read)"
            with state.lock:
                state.tracker.mark_read(r, text)
            role = "box" if r.kind == "box" else r.item.role
            print(f"  [{role}] {time.time() - t:.1f}s  {text}", flush=True)
        time.sleep(0.05 if ready else 0.25)
    print("[live] stopped", flush=True)


def vision_loop(state):
    """V key: the combined pipeline. Updates the board picture ~3 times a second; when the board
    has changed and settled, Qwen3-VL analyses it (this thread waits; the video keeps running)."""
    from board_reader import BoardReader
    person_mask(state, state.latest)                      # make sure the person finder exists
    state.reader = BoardReader(state.engine, state.persons)
    print("[vision] started: keep the camera still; the current board is the 'before' picture", flush=True)
    while state.vision:
        frame = sharpest(state)
        if frame is None:
            time.sleep(0.1)
            continue
        now = time.time()
        state.reader.update(frame.copy(), now)
        if state.reader.should_analyse(now):
            print("[vision] board changed: analysing...", flush=True)
            a = state.reader.analyse(now)
            print(f"[vision] {a.seconds:.1f}s: kept {len(a.kept)}, dropped {len(a.dropped)}", flush=True)
            for it in a.kept:
                what = (f"{'DEFINE' if it.has_content else 'CALL'} artifact, label: {it.label}" if it.kind == "box"
                        else f"[{it.role or it.kind}] {it.text}")
                print(f"  {what}", flush=True)
            for it, why in a.dropped:
                print(f"  (dropped {it.kind} {(it.label or it.text)!r}: {why})", flush=True)
        time.sleep(0.3)
    print("[vision] stopped", flush=True)


def draw_vision(frame, state):
    out = frame.copy()
    reader = state.reader
    if reader is None:
        return out
    t = max(2, out.shape[1] // 600)
    colours = {"box": (0, 170, 0), "autocomplete": (0, 0, 230), "graph": (200, 0, 200), "axes": (255, 0, 255),
               "writing": (0, 140, 255), "text": (0, 140, 255), "expression": (0, 140, 255)}
    with reader.lock:
        items = list(reader.items)
    for it in items:
        key = it.role if it.kind == "expression" and it.role else it.kind
        c = colours.get(key, (0, 140, 255))
        x0, y0, x1, y1 = it.bbox
        cv2.rectangle(out, (x0, y0), (x1, y1), c, t)
        label = (f"{'DEFINE' if it.has_content else 'CALL'}: {it.label}" if it.kind == "box"
                 else ("axes" if it.kind == "axes" else f"{key}: {it.text}"))
        cv2.putText(out, label[:60], (x0, max(12, y0 - 3 * t)), cv2.FONT_HERSHEY_SIMPLEX, t * 0.4, c, t)
    return out


def draw_live(frame, state):
    """Live mode overlay: grey = waiting (writing in progress / covered), coloured = read."""
    with state.lock:
        regions = list(state.tracker.regions) if state.tracker else []
        person = state.person_mask
    out = frame.copy()
    if person is not None and person.shape[:2] == out.shape[:2]:
        out[person > 0] = (0.6 * out[person > 0] + [60, 0, 60]).astype(np.uint8)   # people: purple tint
    t = max(2, out.shape[1] // 600)
    colours = {"box": (0, 170, 0), "autocomplete": (0, 0, 230), "graph": (200, 0, 200), "writing": (0, 140, 255),
               "axes": (255, 0, 255)}
    for r in regions:
        role = r.kind if r.kind != "writing" else r.item.role
        colour = colours.get(role, (0, 140, 255)) if (r.reading or r.kind == "axes") else (150, 150, 150)
        x0, y0, x1, y1 = r.bbox
        if r.kind == "box":
            cv2.polylines(out, [r.item.corners.astype(np.int32)], True, colour, t * 2)
        else:
            cv2.rectangle(out, (x0, y0), (x1, y1), colour, t)
        label = r.reading if r.reading else ("covered" if r.covered else "waiting...")
        if r.kind != "axes":
            cv2.putText(out, f"{role}: {label}"[:70], (x0, max(12, y0 - 3 * t)), cv2.FONT_HERSHEY_SIMPLEX,
                        t * 0.4, colour, t)
    return out


def snip_and_read(state, frame):
    """S key: find every region on the board, then read them one by one in the background."""
    from snippet import snip_board

    def job():
        t0 = time.time()
        boxes, axes, writing = snip_board(frame, ignore=person_mask(state, frame))
        with state.lock:
            state.regions, state.readings = (boxes, axes, writing), {}
        print(f"[snip] {len(boxes)} box(es), {len(axes)} graph axes, {len(writing)} writing snippet(s) "
              f"in {time.time() - t0:.2f}s", flush=True)
        log = []
        for i, b in enumerate(boxes):
            label = state.engine.recognize(b.label_crop, hint="text") or "(no label read)"
            text = f"{'DEFINE' if b.has_content else 'CALL'} artifact, label: {label}"
            with state.lock:
                state.readings[("box", i)] = text
            print(f"  box {i + 1}: {text}", flush=True)
            log.append({"kind": "box", "bbox": b.bbox, "result": text})
        for i, s in enumerate(writing):
            t = time.time()
            text = state.engine.recognize(s.crop, hint=s.hint) or "(nothing read)"
            with state.lock:
                state.readings[("writing", i)] = text
            print(f"  {i + 1}: [{s.role}] {time.time() - t:.1f}s  {text}", flush=True)
            log.append({"kind": s.role, "bbox": s.bbox, "result": text})
        for a in axes:
            log.append({"kind": "axes", "bbox": a.bbox})
        save_frame(frame, log)
        with state.lock:
            state.busy = False
        print(f"[snip] done in {time.time() - t0:.1f}s", flush=True)

    state.busy, state.preview = True, False
    threading.Thread(target=job, daemon=True).start()


def save_frame(frame, regions):
    CAPTURES.mkdir(exist_ok=True)
    name = time.strftime("%Y%m%d-%H%M%S") + "_board.png"
    cv2.imwrite(str(CAPTURES / name), frame)
    with open(CAPTURES / "log.jsonl", "a") as f:
        f.write(json.dumps({"image": name, "mode": "snip", "regions": regions}, default=int) + "\n")


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


def draw_regions(frame, state):
    """Draw the snippeter's regions (and what was read in them) on the full-size frame."""
    from snippet import draw_snippets
    with state.lock:
        regions, readings = state.regions, dict(state.readings)
    if not regions:
        return frame
    boxes, axes, writing = regions
    out = draw_snippets(frame, boxes, axes, writing, [readings.get(("writing", i)) for i in range(len(writing))])
    t = max(2, out.shape[1] // 600)
    for i, b in enumerate(boxes):
        if ("box", i) in readings:
            x0, y0, x1, y1 = b.bbox
            cv2.putText(out, readings[("box", i)][:60], (x0, y1 + 12 * t), cv2.FONT_HERSHEY_SIMPLEX, t * 0.45,
                        (0, 170, 0), t)
    return out


def draw(view, state, scale, fps):
    h, w = view.shape[:2]
    if state.box:
        x0, y0, x1, y1 = (int(v * scale) for v in state.box)
        cv2.rectangle(view, (x0, y0), (x1, y1), (0, 200, 255), 2)
    lines = [state.status if state.engine is None else
             (("VISION MODE: analysing the board..." if state.reader and state.reader.busy else
               "VISION MODE: waiting for the board to change and settle (V to stop)") if state.vision else
              "LIVE MODE: reading regions once they are still (R to stop)" if state.live else
              ("Reading..." if state.busy else
               ("Snippet preview ON (D to stop, S to read everything)" if state.preview else
                "R = live mode, S = read the whole board once, D = show snippets, or drag a box + M/T/A")))]
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

            state.latest = frame
            with state.lock:                     # keep ~0.5 s of frames to pick the sharpest from
                state.recent = (state.recent + [(sharpness(frame), frame)])[-15:]
            shown = (draw_vision(frame, state) if state.vision else
                     draw_live(frame, state) if state.live else draw_regions(frame, state))
            view = cv2.resize(shown, None, fx=scale, fy=scale) if scale < 1 else shown.copy()
            draw(view, state, scale, fps)
            cv2.imshow(WINDOW, view)
            if vcam:
                vcam.send(view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                state.box, state.result = None, None
                with state.lock:
                    state.regions, state.readings, state.preview = None, {}, False
            if key == ord("d"):
                state.preview = not state.preview
                if state.preview:
                    threading.Thread(target=preview_loop, args=(state,), daemon=True).start()
                else:
                    with state.lock:
                        state.regions = None
            if key == ord("s") and state.engine is not None and not state.busy and not state.live:
                snip_and_read(state, sharpest(state).copy())
            if key == ord("v") and state.engine is not None and not state.busy and not state.live:
                state.vision = not state.vision
                if state.vision:
                    state.preview = False
                    with state.lock:
                        state.regions = None
                    threading.Thread(target=vision_loop, args=(state,), daemon=True).start()
            if key == ord("r") and state.engine is not None and not state.busy and not state.vision:
                state.live = not state.live
                if state.live:
                    state.preview = False
                    with state.lock:
                        state.regions = None
                    threading.Thread(target=live_loop, args=(state,), daemon=True).start()
            ready = state.engine is not None and not state.busy and crop is not None
            if key in (ord("m"), ord("t"), ord("a"), ord("l")) and ready:
                mode = {ord("m"): "math", ord("t"): "text", ord("a"): "auto", ord("l"): "label"}[key]
                read_in_background(state, [crop], mode, labels)
            if key == ord("f") and ready:
                # three frames spread over the last ~0.5 s of video
                picks = frames_for_vote[::max(1, len(frames_for_vote) // 3)][-3:] or [crop]
                read_in_background(state, picks, "frames", labels)
    finally:
        state.live = state.preview = state.vision = False
        cap.release()
        if vcam:
            vcam.close()
        cv2.destroyAllWindows()
        time.sleep(0.3)                     # let the live thread finish its current step
        if state.persons is not None:
            state.persons.close()


if __name__ == "__main__":
    main()
