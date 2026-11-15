"""The combined pipeline: what is new on the board, understood by Qwen3-VL.

    reader = BoardReader(engine, PersonFinder())
    for each camera frame (a few per second):
        reader.update(frame, now)          # cheap: keeps a person-free picture of the board
        if reader.should_analyse(now):     # the board changed and has settled
            reader.analyse(now)            # slow (~15-40 s): run it in a background thread
        reader.items                       # what is on the board now, with readings

How it works (chosen by testing on photos and a recorded lecture, see tests/board):
  1. Board picture: the per-pixel median of the last few seconds, updated only where no person
     stands (NewInkTracker). The start of the session is the BASELINE.
  2. When the board has changed and settled (nobody writing for STABLE_SECONDS), Qwen3-VL looks
     at the whole board picture and lists boxes (label, empty or not), expressions, text, axes.
  3. Every item is checked against the baseline: only items whose ink is mostly NEW are kept.
     This drops old writing, clocks, screens and anything the model invented (no ink there).
  4. Expressions and text are re-read from their crop (crop reading was more accurate);
     the model's own reading is the fallback. Box labels come from the model.
Why not the alternatives: snippet.py's rules missed faint boxes and read clutter as writing;
triggering on every frame's new ink was too noisy for faint marker; Qwen3-VL alone also listed
old writing and occasionally invented items, which step 3 removes.
"""
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

from newink import NewInkTracker
from ocr_engine import ink_mask, vote
from vlm_layout import find_with_vlm

STABLE_SECONDS = 1.5      # the board must be unchanged this long before analysing
MIN_CHANGE = 120          # pixels of ink that appeared/disappeared since the last analysis
MIN_INTERVAL = 8.0        # seconds between analyses (each takes 15-40 s anyway)
NEW_SHARE = 0.4           # keep an item if at least 40% of its ink is new since the start


@dataclass
class BoardItem:
    kind: str                 # "box", "expression", "text" or "axes"
    bbox: tuple               # frame pixels (x0, y0, x1, y1)
    role: str = ""            # expressions: "autocomplete" / "graph" / "writing"
    text: str = ""            # reading (LaTeX for expressions)
    label: str = ""           # boxes: the label written in the corner
    has_content: bool = False # boxes: True -> DEFINE artifact, False -> CALL artifact
    new_share: float = 1.0    # share of its ink that is new since the start
    since: float = 0.0        # when it was first found
    readings: list = field(default_factory=list)   # every reading of it so far (they vote)


@dataclass
class Analysis:
    time: float
    seconds: float
    kept: list = field(default_factory=list)
    dropped: list = field(default_factory=list)   # (item, reason)
    region: tuple = None                          # the area that was analysed (frame pixels)


class BoardReader:
    def __init__(self, engine, person_finder, all_new=False):
        self.engine, self.people = engine, person_finder
        self.ink = NewInkTracker(all_new=all_new)
        self.items = []
        self.history = []          # list of Analysis
        self.busy = False
        self._analysed_new = None  # new-ink mask at the last analysis
        self._last_change = 0.0
        self._last_analysis = -1e9
        self._prev_new = None
        self.lock = threading.Lock()

    # ---------- cheap, every update ----------
    def update(self, frame_bgr, now):
        self.ink.step(frame_bgr, self.people.mask(frame_bgr), now)
        if self.ink.board is None:
            return
        new = self._new_ink()
        if self._prev_new is not None and _changed(self._prev_new, new) > MIN_CHANGE / 2:
            self._last_change = now           # someone is still writing / erasing
        self._prev_new = new

    def waiting_for(self, now):
        """Why no analysis is happening right now, in plain words (for the log)."""
        if self.ink.board is None:
            return "a clear view of the board (someone is in view, or a person is detected by mistake)"
        if self.busy:
            return "the current analysis to finish"
        new = self._new_ink()
        if now - self._last_change < STABLE_SECONDS:
            return "the board to stay still (ink is still changing)"
        n = int((new > 0).sum()) if self._analysed_new is None else int(_changed(self._analysed_new, new))
        if n < MIN_CHANGE:
            return (f"new writing ({n} new ink pixels, needs {MIN_CHANGE}); anything already on the board "
                    f"when the app started or B was pressed is ignored - press N to count it as new")
        return "the next analysis slot"

    def should_analyse(self, now):
        if self.busy or self.ink.board is None or now - self._last_analysis < MIN_INTERVAL:
            return False
        if now - self._last_change < STABLE_SECONDS:
            return False
        new = self._new_ink()
        if self._analysed_new is None:
            return (new > 0).sum() >= MIN_CHANGE          # first writing appeared
        return _changed(self._analysed_new, new) >= MIN_CHANGE

    # ---------- slow, when the board changed ----------
    def analyse(self, now):
        """Find and read what is new. Only the area that changed since the last analysis is sent
        to Qwen3-VL (much faster than the whole board); items elsewhere are kept as they were."""
        import time
        self.busy = True
        t0 = time.time()
        try:
            board = self.ink.board.copy()
            new_mask = self._new_ink()
            scale = self.ink.scale
            full = cv2.resize(board, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_CUBIC)
            region = self._changed_region(new_mask, scale)           # (x0, y0, x1, y1) in board pixels
            rx0, ry0, rx1, ry1 = region
            boxes, axes, writing = find_with_vlm(self.engine, board[ry0:ry1, rx0:rx1])
            shift = lambda bb: (bb[0] + rx0, bb[1] + ry0, bb[2] + rx0, bb[3] + ry0)
            analysis = Analysis(now, 0.0)
            analysis.region = tuple(int(v / scale) for v in region)
            found = []
            for bx in boxes:
                found.append(BoardItem("box", shift(bx.bbox), label=bx.text, has_content=bx.has_content))
            for ax in axes:
                found.append(BoardItem("axes", shift(ax.bbox)))
            for s in writing:
                found.append(BoardItem("expression" if s.hint == "math" else "text", shift(s.bbox),
                                       role=s.role, text=s.text))
            kept = []
            for it in found:
                share, ink_px = self._new_share(it.bbox, board, new_mask)
                it.new_share = share
                if ink_px == 0:
                    analysis.dropped.append((it, "no ink there (invented)"))
                elif share < NEW_SHARE:
                    analysis.dropped.append((it, f"already on the board at the start ({share:.0%} new)"))
                else:
                    kept.append(it)
            for it in kept:
                if it.kind == "box":                         # DEFINE vs CALL: measured, not guessed
                    it.has_content = self._new_ink_inside(it.bbox, new_mask)
            for it in kept:                                  # re-read writing from its crop
                it.bbox = tuple(int(v / scale) for v in it.bbox)
                if it.kind in ("expression", "text"):
                    x0, y0, x1, y1 = it.bbox
                    crop = full[y0:y1, x0:x1]
                    if crop.size:
                        read = self.engine.recognize(crop, hint="math" if it.kind == "expression" else "text")
                        it.text = read or it.text
                prev = next((p for p in self.items if p.kind == it.kind and _iou(p.bbox, it.bbox) > 0.3), None)
                if prev is not None and self._rewritten(it.bbox, new_mask, scale):
                    prev = None                # erased and rewritten ("2+2=" -> "9+10="): old readings don't vote
                it.since = prev.since if prev else now
                # Re-reads of the same item vote: the most frequent reading wins, so one bad read
                # of a faint label ("grape" after several "apple") doesn't overwrite a good one.
                it.readings = (prev.readings if prev else []) + [it.label if it.kind == "box" else it.text]
                best = vote(it.readings)
                if it.kind == "box":
                    it.label = best or it.label
                elif it.kind in ("expression", "text"):
                    it.text = best or it.text
            # A label or drawing inside a box belongs to the box, not a separate item
            boxes_now = [b for b in kept if b.kind == "box"]
            kept = [it for it in kept if it.kind not in ("expression", "text") or
                    not any(_inside_share(it.bbox, b.bbox) > 0.7 for b in boxes_now)]
            # Earlier items stay until their ink is erased or a new reading of the same kind
            # replaces them: the model doesn't list every item on every run, and an item that
            # flickered out (a box, the axes) would otherwise vanish while still on the board.
            stay = [p for p in self.items if self._still_there(p, new_mask, scale) and
                    not any(n.kind == p.kind and _iou(n.bbox, p.bbox) > 0.3 for n in kept)]
            analysis.kept = kept
            analysis.seconds = time.time() - t0
            with self.lock:
                self.items = stay + kept
                self.history.append(analysis)
                self._analysed_new = new_mask
                self._last_analysis = now
            return analysis
        finally:
            self.busy = False

    def _changed_region(self, new_mask, scale, margin=0.12):
        """The area to analyse: where ink appeared or vanished since the last analysis, with a
        margin, widened to include whole items it touches. The whole board on the first
        analysis or when the change is spread over most of it."""
        h, w = new_mask.shape
        whole = (0, 0, w, h)
        if self._analysed_new is None:
            return whole
        k = np.ones((7, 7), np.uint8)
        changed = (((new_mask > 0) & (cv2.dilate(self._analysed_new, k) == 0)) |
                   ((self._analysed_new > 0) & (cv2.dilate(new_mask, k) == 0))).astype(np.uint8)
        # only substantial changes: faint ink flickering in and out leaves small scattered specks
        # all over the board, and their bounding box was always the whole board
        grouped = cv2.dilate(changed, np.ones((15, 15), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, 8)
        big = [i for i in range(1, n) if (changed[labels == i] > 0).sum() >= MIN_CHANGE / 3]
        changed = np.isin(labels, big) & (changed > 0) if big else np.zeros_like(changed, bool)
        ys, xs = np.nonzero(changed)
        if len(xs) == 0:
            return whole
        m = int(min(h, w) * margin)
        x0, y0, x1, y1 = max(0, xs.min() - m), max(0, ys.min() - m), min(w, xs.max() + m), min(h, ys.max() + m)
        grown = True
        while grown:                                   # take in whole items the area touches
            grown = False
            for p in self.items:
                px0, py0, px1, py1 = (int(v * scale) for v in p.bbox)
                if px0 < x1 and x0 < px1 and py0 < y1 and y0 < py1:
                    nx0, ny0, nx1, ny1 = min(x0, px0), min(y0, py0), max(x1, px1), max(y1, py1)
                    if (nx0, ny0, nx1, ny1) != (x0, y0, x1, y1):
                        x0, y0, x1, y1, grown = nx0, ny0, nx1, ny1, True
        if (x1 - x0) * (y1 - y0) > 0.6 * w * h:
            return whole
        return int(x0), int(y0), int(x1), int(y1)

    def _rewritten(self, bbox, new_mask, scale, share=0.3):
        """Did the ink in this area change a lot since the last analysis? (Small changes, like
        flicker or a touched-up stroke, keep the earlier readings voting.)"""
        if self._analysed_new is None:
            return False
        x0, y0, x1, y1 = (int(v * scale) for v in bbox)
        now, before = new_mask[y0:y1, x0:x1] > 0, self._analysed_new[y0:y1, x0:x1] > 0
        k = np.ones((5, 5), np.uint8)
        gone = before & (cv2.dilate(now.astype(np.uint8), k) == 0)
        added = now & (cv2.dilate(before.astype(np.uint8), k) == 0)
        ink = max(int(now.sum()), int(before.sum()), 1)
        return (gone.sum() + added.sum()) / ink >= share

    def _still_there(self, item, new_mask, scale):
        """An item outside the analysed area stays unless its new ink was erased."""
        x0, y0, x1, y1 = (int(v * scale) for v in item.bbox)
        return int((new_mask[y0:y1, x0:x1] > 0).sum()) >= 15

    # ---------- helpers ----------
    def _new_ink(self):
        return ((self.ink.persist > 0.6) * 255).astype(np.uint8) if self.ink.persist is not None else None

    @staticmethod
    def _new_ink_inside(bbox, new_mask, edge=0.15, min_share=0.004):
        """Is anything new drawn inside the box, besides its outline and its label? The middle of
        the box (edges and the bottom-left label corner left out) must hold some new ink.
        Qwen3-VL's own guess called an empty box DEFINE on the recorded lecture."""
        x0, y0, x1, y1 = bbox
        w, h = x1 - x0, y1 - y0
        inner = new_mask[y0 + int(h * edge): y1 - int(h * edge), x0 + int(w * edge): x1 - int(w * edge)].copy()
        if inner.size == 0:
            return False
        ih, iw = inner.shape
        inner[int(ih * 0.55):, :int(iw * 0.6)] = 0          # the label corner
        return float((inner > 0).mean()) >= min_share

    def _new_share(self, bbox, board, new_mask):
        """Share of the ink in this area that is new since the start, and how much ink there is."""
        x0, y0, x1, y1 = bbox
        region = board[y0:y1, x0:x1]
        if region.size == 0:
            return 0.0, 0
        ink = ink_mask(region, board=self.ink.board_kind) > 0
        old = self.ink.baseline_ink[y0:y1, x0:x1] > 0
        total = int(ink.sum())
        return (float((ink & ~old).sum()) / total if total else 0.0), total


def _changed(a, b):
    """Ink pixels that appeared or disappeared (ignoring shifts of a few pixels)."""
    if a is None or b is None:
        return 0
    k = np.ones((7, 7), np.uint8)
    appeared = (b > 0) & (cv2.dilate(a, k) == 0)
    vanished = (a > 0) & (cv2.dilate(b, k) == 0)
    return int(appeared.sum() + vanished.sum())


def _inside_share(inner, outer):
    """Share of `inner`'s area that lies inside `outer`."""
    area = max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return _iou_any(inner, outer) / area


def _iou_any(a, b):
    """Overlap area of two (x0, y0, x1, y1) boxes (0 = they don't touch)."""
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
