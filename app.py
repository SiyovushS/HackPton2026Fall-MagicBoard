"""Magic Board Cam: the prototype.

    .venv/bin/python app.py                            # webcam 0 -> window + OBS Virtual Camera (Zoom)
    .venv/bin/python app.py --no-vcam                  # window only
    .venv/bin/python app.py --video recordings/board_test.mov --out recordings/board_test_magic.mp4

The board when the app starts is the "before" picture: only what is written after that is used.
Boxes showing a saved drawing are tracked frame by frame (Darcy's method), so the drawing follows
the box when the camera moves. While running:
    B   take a new "before" picture yourself
    N   count everything on the board right now as new (e.g. boxes drawn before pressing B)
    D   show/hide outlines of everything found (debug)
    Q   quit

What appears on the board (drawn behind the professor):
    2 + 2 = ___                    the answer after the "="
    x^2 + 2x + 3  next to axes     the curve, plotted inside the drawn axes
    box + drawing + label          DEFINE: the drawing is saved under the label (artifacts/)
    empty box + label              CALL: the saved drawing appears inside the box
"""
import argparse
import threading
import time

import cv2
import numpy as np

from actions import ArtifactStore, graph_points, solve_autocomplete, _to_sympy
from board_reader import BoardReader
from box_quad import BoxTrack, find_quad, paste_into_quad, rectify
from ocr_engine import OCREngine, has_ink, clean_latex
from people import PersonFinder
from snippet import LABEL_H, LABEL_W

INK = (255, 255, 255)        # BGR: the colour answers and curves are drawn in (white, like chalk)


class MagicBoard:
    def __init__(self, engine, people, store):
        self.engine, self.people, self.store = engine, people, store
        self.reader = BoardReader(engine, people)
        self.lock = threading.Lock()
        self.answers = {}        # expression text -> answer
        self.curves = {}         # (expression text, axes bbox) -> points
        self.saved = set()       # labels saved as artifacts in this session
        self.beside = {}         # axes bbox -> expression read next to it (fallback)
        self.quads = {}          # box bbox -> its exact 4 corners (Darcy's box detection)
        self.tracks = []         # [label, BoxTrack]: CALL boxes followed frame by frame (Darcy's)
        self.debug = False
        self._quiet_since = None

    def reset(self, all_new=False):
        """Start over. all_new=True: what is on the board now counts as new writing too."""
        with self.lock:
            self.reader = BoardReader(self.engine, self.people, all_new=all_new)
            self._quiet_since = None
            self.answers, self.curves, self.beside, self.quads, self.tracks = {}, {}, {}, {}, []

    # ---------- slow side: a few times a second, plus analyses ----------
    def update(self, frame, now, log=print):
        reader = self.reader
        reader.update(frame, now)
        if not reader.should_analyse(now):
            if self._quiet_since is None:
                self._quiet_since = now
            elif now - self._quiet_since >= 10:          # say why nothing is happening
                log(f"[waiting] for {reader.waiting_for(now)}")
                self._quiet_since = now
            return
        self._quiet_since = None
        a = reader.analyse(now)
        log(f"[board] analysed in {a.seconds:.1f}s: {len(a.kept)} new/updated, {len(a.dropped)} ignored")
        items = list(reader.items)
        board = None
        for it in items:                       # snap each box to its real outline (once per position)
            if it.kind == "box" and it.bbox not in self.quads:
                board = board if board is not None else self._full_board()
                self.quads[it.bbox] = find_quad(board, it.bbox)
        for it in items:
            if it.kind == "box" and it.has_content and it.label:
                inside, new_ink = self._box_inside(it)
                if self.store.save(it.label, inside, keep=new_ink) is not None and it.label not in self.saved:
                    self.saved.add(it.label)
                    log(f"  artifact saved: '{it.label}'")
            elif it.kind == "box" and not it.has_content and it.label and self.quads.get(it.bbox) is not None:
                self._track_call_box(it.label, self.quads[it.bbox])
            elif it.kind == "expression" and it.role == "autocomplete" and it.text not in self.answers:
                self.answers[it.text] = solve_autocomplete(it.text)
                log(f"  {it.text}  ->  {self.answers[it.text]}")
        live = set()
        pairs = _graph_pairs(items)
        paired = {id(ax) for _, ax in pairs}
        for ax in items:                       # axes with nothing beside them: read the ink left of the arm
            if ax.kind == "axes" and id(ax) not in paired:
                it = self._expression_beside(ax)
                if it is not None:
                    pairs.append((it, ax))
        for it, axes in pairs:
            key = (it.text, axes.bbox)
            live.add(key)
            if key not in self.curves:
                corner, arms = _axes_corner(self._full_board(), axes.bbox)
                self.curves[key] = graph_points(it.text, arms, corner=corner)
                log(f"  graph of {it.text}: {'drawn' if self.curves[key] is not None else 'cannot plot'}")
        for key in list(self.curves):          # expression or axes erased: drop the curve
            if key not in live:
                del self.curves[key]

    def _track_call_box(self, label, quad):
        """Start following a CALL box, unless it's already followed (then just update its label).
        A track lives on its own: it isn't dropped when an analysis loses the box (e.g. after the
        camera moved); only Darcy's confidence count decides when the box is gone."""
        with self.lock:
            for tr in self.tracks:
                if _quad_iou(tr[1].pos, quad) > 0.3:
                    tr[0] = label
                    return
            self.tracks.append([label, BoxTrack(quad)])

    def _expression_beside(self, axes):
        """Fallback when the layout model lists the axes but not the expression next to them
        (it sometimes counts the expression as part of the axes): read the ink left of the
        vertical arm, level with the axes. Remembered once read successfully."""
        if axes.bbox in self.beside:
            return self.beside[axes.bbox]
        board = self._full_board()
        corner, arms = _axes_corner(board, axes.bbox)
        if corner is None:
            return None
        ax0, ay0, ax1, ay1 = arms
        w = ax1 - ax0
        x0 = max(0, int(ax0 - 1.0 * w))
        x1 = int(ax0 - 0.04 * w)              # stop before the arm (and its arrowhead)
        crop = board[ay0:ay1, x0:x1]
        if x1 - x0 < 10 or not has_ink(crop):
            return None
        text = self.engine.recognize(crop, hint="math")
        if text and _graphable(text):
            from board_reader import BoardItem
            item = BoardItem("expression", (x0, ay0, x1, ay1), role="graph", text=text)
            self.beside[axes.bbox] = item
            return item
        return None

    def _full_board(self):
        """The person-free board picture at frame size."""
        ink = self.reader.ink
        size = (int(ink.board.shape[1] / ink.scale), int(ink.board.shape[0] / ink.scale))
        return cv2.resize(ink.board, size, interpolation=cv2.INTER_CUBIC)

    def _box_inside(self, item):
        """The inside of a DEFINE box (outline and label corner left out), from the person-free
        board, plus a mask of the ink in it that is NEW since the session started."""
        ink = self.reader.ink
        full = self._full_board()
        size = (full.shape[1], full.shape[0])
        new = cv2.resize(((ink.persist > 0.6) * 255).astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
        quad = self.quads.get(item.bbox)
        if quad is not None:                   # exact outline: straighten the inside, just within the lines
            inside, new_ink = rectify(full, quad, inset=0.06), rectify(new, quad, inset=0.06)
        else:                                  # rough box from the layout model
            x0, y0, x1, y1 = item.bbox
            w, h = x1 - x0, y1 - y0
            sl = (slice(y0 + int(h * 0.08), y1 - int(h * 0.08)), slice(x0 + int(w * 0.08), x1 - int(w * 0.08)))
            inside, new_ink = full[sl].copy(), new[sl].copy()
        if inside is None or inside.size == 0:
            return None, None
        inside, new_ink = inside.copy(), new_ink.copy()
        ih, iw = inside.shape[:2]
        inside[int(ih * (1 - LABEL_H)):, :int(iw * LABEL_W)] = np.median(inside.reshape(-1, 3), axis=0)
        new_ink[int(ih * (1 - LABEL_H)):, :int(iw * LABEL_W)] = 0
        return inside, new_ink

    # ---------- fast side: every frame ----------
    def render(self, frame, person=None):
        items = list(self.reader.items)
        layer = frame.copy()
        for it in items:
            if it.kind == "expression" and it.role == "autocomplete":
                answer = self.answers.get(it.text)
                if answer:
                    _write_answer(layer, answer, it.bbox)
            elif it.kind == "box" and not it.has_content and it.label and self.quads.get(it.bbox) is None:
                art = self.store.load(it.label)              # no exact outline found: rough box
                if art is not None:
                    _paste_artifact(layer, art, it.bbox)
        with self.lock:
            tracks = list(self.tracks)
        for tr in tracks:                                    # CALL boxes, followed every frame
            label, track = tr
            quad = track.step(frame, person)
            if quad is None:                                 # erased (or lost for good)
                with self.lock:
                    if tr in self.tracks:
                        self.tracks.remove(tr)
                continue
            art = self.store.load(label)
            if art is not None:
                paste_into_quad(layer, art, quad)            # warped into the box's own corners
        t = max(2, frame.shape[1] // 400)
        for points in self.curves.values():
            if points is not None and len(points) > 1:
                cv2.polylines(layer, [points.reshape(-1, 1, 2)], False, INK, t, cv2.LINE_AA)
        if self.debug:
            for it in items:
                cv2.rectangle(layer, it.bbox[:2], it.bbox[2:], (0, 200, 0), 1)
                cv2.putText(layer, f"{it.kind} {it.label or it.text}"[:40], (it.bbox[0], max(10, it.bbox[1] - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        if person is not None:                 # behind the professor: their pixels stay on top
            layer[person > 0] = frame[person > 0]
        return layer


def _graph_pairs(items):
    """(expression, axes) pairs to plot. The expression is written left of the axes' vertical
    arm and level with them. Layout boxes can be loose (an expression's box may swallow most of
    the axes), so only its LEFT edge has to be before the arm, not the whole box."""
    axes_list = [x for x in items if x.kind == "axes"]
    pairs = []
    for it in items:
        if it.kind not in ("expression", "text") or it.role == "autocomplete" or not it.text:
            continue
        if it.text.rstrip().endswith("=") or not _graphable(it.text):
            continue                           # e.g. "2 \\times": the fallback reads the ink instead
        best = None
        for ax in axes_list:
            x0, y0, x1, y1 = it.bbox
            ax0, ay0, ax1, ay1 = ax.bbox
            aw = ax1 - ax0
            level = min(y1, ay1) - max(y0, ay0) > 0.3 * min(y1 - y0, ay1 - ay0)
            left = ax0 - 0.8 * aw <= x0 <= ax0 + 0.25 * aw
            if level and left:
                d = abs(x0 - ax0)
                if best is None or d < best[0]:
                    best = (d, ax)
        if best:
            pairs.append((it, best[1]))
    return pairs


def _quad_iou(a, b):
    """Overlap of two quads' bounding rectangles."""
    ax0, ay0 = a.min(axis=0); ax1, ay1 = a.max(axis=0)
    bx0, by0 = b.min(axis=0); bx1, by1 = b.max(axis=0)
    iw, ih = max(0, min(ax1, bx1) - max(ax0, bx0)), max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    return inter / ((ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter)


def _graphable(latex):
    """True if the reading is a function of one variable (so a curve can be drawn)."""
    import re
    expr = _to_sympy(re.sub(r"^\s*y\s*=\s*", "", clean_latex(latex)))
    return expr is not None and len(expr.free_symbols) == 1


def _axes_corner(board, bbox):
    """Where the drawn L's arms meet, and the box the arms really span: (corner, bbox).
    The vertical arm is the strongest ink column, the horizontal arm the strongest ink row.
    (None, bbox) if no clear L is found."""
    from ocr_engine import ink_mask
    x0, y0, x1, y1 = bbox
    crop = board[max(0, y0):y1, max(0, x0):x1]
    if crop.size == 0:
        return None, bbox
    ink = ink_mask(crop) > 0
    k = max(3, min(ink.shape) // 40) | 1                     # hand-drawn arms are a bit slanted
    cols = np.convolve(ink.sum(axis=0), np.ones(k), "same")
    rows = np.convolve(ink.sum(axis=1), np.ones(k), "same")
    if cols.max() < 0.3 * ink.shape[0] or rows.max() < 0.3 * ink.shape[1]:   # no clear L
        return None, bbox
    # the vertical arm is the leftmost strong column, the horizontal arm the lowest strong row
    cx = int(np.nonzero(cols >= 0.7 * cols.max())[0][0])
    cy = int(np.nonzero(rows >= 0.7 * rows.max())[0][-1])
    # how far the arms really reach (the layout box can be bigger than the drawing)
    band = max(2, k)
    up = np.nonzero(ink[:cy, max(0, cx - band):cx + band + 1].any(axis=1))[0]
    right = np.nonzero(ink[max(0, cy - band):cy + band + 1, cx:].any(axis=0))[0]
    top = int(up[0]) if len(up) else 0
    end = cx + int(right[-1]) if len(right) else ink.shape[1]
    ox, oy = max(0, x0), max(0, y0)
    return (ox + cx, oy + cy), (ox + cx, oy + top, ox + end, oy + cy)


def _write_answer(img, answer, bbox):
    x0, y0, x1, y1 = bbox
    h = max(16, y1 - y0)
    scale = h / 40
    text = answer.replace("\\approx", "~").replace("\\", "")
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, max(2, int(scale * 2)))
    org = (x1 + int(h * 0.2), y0 + (y1 - y0 + th) // 2)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, INK, max(2, int(scale * 2)), cv2.LINE_AA)


def _paste_artifact(img, art_bgra, bbox, inset=0.12):
    """Draw a saved drawing inside a box, keeping its proportions, blended by its transparency."""
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    bx0, by0, bw, bh = x0 + int(w * inset), y0 + int(h * inset), int(w * (1 - 2 * inset)), int(h * (1 - 2 * inset))
    ah, aw = art_bgra.shape[:2]
    s = min(bw / aw, bh / ah)
    if s <= 0:
        return
    art = cv2.resize(art_bgra, (max(1, int(aw * s)), max(1, int(ah * s))), interpolation=cv2.INTER_AREA)
    ox, oy = bx0 + (bw - art.shape[1]) // 2, by0 + (bh - art.shape[0]) // 2
    H, W = img.shape[:2]
    if ox < 0 or oy < 0 or ox + art.shape[1] > W or oy + art.shape[0] > H:
        return
    alpha = art[..., 3:4].astype(np.float32) / 255
    region = img[oy:oy + art.shape[0], ox:ox + art.shape[1]].astype(np.float32)
    img[oy:oy + art.shape[0], ox:ox + art.shape[1]] = (alpha * art[..., :3] + (1 - alpha) * region).astype(np.uint8)


# ---------------- running it ----------------
def run_video(app, path, out_path, show):
    """Offline: play a recording through the app and save what Zoom would show."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer, i, step = None, 0, max(1, int(round(fps / 3)))
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = i / fps
        if i % step == 0:
            app.update(frame, now, log=lambda m: print(f"t={now:6.1f}s {m}", flush=True))
        out = app.render(frame, app.people.mask(frame))
        if out_path:
            if writer is None:
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
            writer.write(out)
        if show:
            cv2.imshow("Magic Board (video)", cv2.resize(out, None, fx=0.6, fy=0.6))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        i += 1
    if writer:
        writer.release()
    print(f"done: {i / fps:.0f}s of video in {time.time() - t0:.0f}s" + (f", saved {out_path}" if out_path else ""))


def run_camera(app, index, vcam_on):
    from live_ocr_test import open_camera, sharpness
    cap = open_camera(index)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not read from the camera (System Settings > Privacy & Security > Camera).")
    h, w = frame.shape[:2]
    latest, recent, running = {"frame": frame}, [], {"on": True}

    def background():                                      # reader updates + analyses
        while running["on"]:
            frames = list(recent) or [(0, latest["frame"])]
            best = max(frames, key=lambda sf: sf[0])[1]   # sharpest recent frame (motion blur)
            try:
                app.update(best.copy(), time.time(), log=lambda m: print(m, flush=True))
            except Exception:                              # log it and keep going: one bad frame
                import traceback                           # mustn't stop all analysis
                traceback.print_exc()
                time.sleep(1)
            time.sleep(0.3)

    threading.Thread(target=background, daemon=True).start()
    vcam = None
    if vcam_on:
        import pyvirtualcam
        vcam = pyvirtualcam.Camera(width=w, height=h, fps=30, fmt=pyvirtualcam.PixelFormat.BGR)
        print(f"Sending to {vcam.device} (choose it as the camera in Zoom)", flush=True)
    print("Running. B = new 'before' picture, N = count what's on the board as new, D = debug outlines, Q = quit", flush=True)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            if frame.shape[:2] != (h, w):
                # cameras can switch resolution after opening (1920x1080 first, then 1280x720):
                # keep one size, which the virtual camera and the board picture both need
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
            latest["frame"] = frame
            recent[:] = (recent + [(sharpness(frame), frame)])[-15:]
            out = app.render(frame, app.people.mask(frame))
            if vcam:
                vcam.send(out)
            cv2.imshow("Magic Board", cv2.resize(out, None, fx=0.6, fy=0.6) if w > 1400 else out)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("b"):
                app.reset()
                print("New 'before' picture taken", flush=True)
            if key == ord("n"):
                app.reset(all_new=True)
                print("Everything on the board now counts as new writing", flush=True)
            if key == ord("d"):
                app.debug = not app.debug
    finally:
        running["on"] = False
        cap.release()
        if vcam:
            vcam.close()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", help="run on a recorded video instead of the camera")
    ap.add_argument("--out", help="with --video: save the result to this video file")
    ap.add_argument("--no-vcam", action="store_true", help="don't send to OBS Virtual Camera")
    ap.add_argument("--no-window", action="store_true", help="with --video: don't show a window")
    ap.add_argument("--debug", action="store_true", help="outline everything found")
    args = ap.parse_args()

    t = time.time()
    engine, people = OCREngine(), PersonFinder()
    app = MagicBoard(engine, people, ArtifactStore("artifacts"))
    app.debug = args.debug
    print(f"Models ready in {time.time() - t:.0f}s ({engine.backend})", flush=True)
    try:
        if args.video:
            run_video(app, args.video, args.out, show=not args.no_window)
        else:
            run_camera(app, args.camera, vcam_on=not args.no_vcam)
    finally:
        people.close()


if __name__ == "__main__":
    main()
