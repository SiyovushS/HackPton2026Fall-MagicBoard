"""Frame-to-frame stability: decide WHEN each region on the board is ready to read.

    tracker = BoardTracker()
    for each processed frame (a few per second):
        person = person_finder.mask(frame)
        regions = snip_board(frame, ignore=person)
        tracker.update(regions, frame, person)
        for r in tracker.ready():              # stable, uncovered, not read yet (or changed)
            r.reading = engine.recognize(...)  # read it once
        tracker.regions                        # everything currently on the board, with readings

A region becomes READY when:
  - it has been seen in most recent frames (a passing hand or a one-frame glitch is ignored),
  - its ink has not changed for STABLE_SECONDS (the professor finished writing it),
  - no person covers it.
It is read once, and re-read only if its ink changes (more writing). It is dropped when it has
not been seen for a while (erased). Boxes are remembered longer, because box detection can
miss a box for a frame or two when the camera moves or a line is faint.
"""
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from ocr_engine import ink_mask

STABLE_SECONDS = 1.5      # ink unchanged this long -> writing is finished
SEEN_SHARE = 0.6          # seen in at least 60% of the updates since it first appeared...
MIN_SEEN = 3              # ...and at least 3 times
FORGET_SECONDS = 3.0      # not seen for this long -> erased or gone
FORGET_BOX_SECONDS = 10.0 # boxes are remembered longer (detection can miss a box for a bit)
CHANGED = 0.25            # ink changed by more than 25% -> new writing, read again
COVERED = 0.05            # a person over more than 5% of the region -> wait


@dataclass
class Tracked:
    kind: str                 # "box", "writing" (role in .item.role), or "axes"
    item: object              # latest snippet.Box / Snippet / Axes
    bbox: tuple               # (x0, y0, x1, y1) in frame pixels
    first_seen: float
    last_seen: float
    stable_since: float       # last time its ink changed
    seen: int = 1
    updates: int = 1          # tracker updates since it first appeared
    signature: np.ndarray = None
    reading: object = None    # what was read, once it was ready
    read_signature: np.ndarray = None
    covered: bool = False
    id: int = field(default=0)


class BoardTracker:
    def __init__(self):
        self.regions = []
        self._next_id = 1

    # ---------- main entry points ----------
    def update(self, snipped, frame_bgr, person_mask=None, now=None):
        """Add one frame's detections (boxes, axes, writing from snippet.snip_board)."""
        now = time.time() if now is None else now
        boxes, axes, writing = snipped
        detections = ([("box", b, b.bbox) for b in boxes] + [("axes", a, a.bbox) for a in axes] +
                      [("writing", s, s.bbox) for s in writing])
        for r in self.regions:
            r.updates += 1
        matched = set()
        for kind, item, bbox in detections:
            r = self._match(kind, bbox, matched)
            sig = _signature(frame_bgr, bbox)
            if r is None:
                r = Tracked(kind, item, bbox, now, now, now, signature=sig, id=self._next_id)
                self._next_id += 1
                self.regions.append(r)
            else:
                if _difference(r.signature, sig) > CHANGED:      # ink changed: writing in progress
                    r.stable_since = now
                r.item, r.bbox, r.signature = item, bbox, sig
                r.last_seen, r.seen = now, r.seen + 1
            matched.add(r.id)
        for r in self.regions:
            r.covered = person_mask is not None and _covered(person_mask, r.bbox) > COVERED
            if r.covered:
                r.last_seen = max(r.last_seen, now - 0.5)   # hidden behind a person: don't forget it
        self.regions = [r for r in self.regions if now - r.last_seen <=
                        (FORGET_BOX_SECONDS if r.kind == "box" else FORGET_SECONDS)]

    def ready(self, now=None):
        """Regions that are stable, uncovered and not read yet (or changed since they were read)."""
        now = time.time() if now is None else now
        out = []
        for r in self.regions:
            if r.kind == "axes" or r.covered:
                continue
            if r.seen < MIN_SEEN or r.seen < SEEN_SHARE * r.updates:
                continue
            if now - r.stable_since < STABLE_SECONDS or now - r.last_seen > 1.0:
                continue
            if r.reading is not None and _difference(r.read_signature, r.signature) <= CHANGED:
                continue
            out.append(r)
        return out

    def mark_read(self, region, reading):
        region.reading, region.read_signature = reading, region.signature

    # ---------- helpers ----------
    def _match(self, kind, bbox, taken):
        best, best_iou = None, 0.3
        for r in self.regions:
            if r.id in taken or r.kind != kind:
                continue
            iou = _iou(r.bbox, bbox)
            if iou > best_iou:
                best, best_iou = r, iou
        return best


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _signature(frame_bgr, bbox, size=(48, 24)):
    """A small picture of the region's ink, to notice when the writing changes."""
    x0, y0, x1, y1 = bbox
    crop = frame_bgr[max(0, y0):y1, max(0, x0):x1]
    if crop.size == 0:
        return np.zeros(size[::-1], np.float32)
    ink = cv2.resize(ink_mask(crop), size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    return cv2.GaussianBlur(ink, (5, 5), 0)    # small camera jitter shouldn't count as a change


def _difference(a, b):
    """How much the ink differs (0 = same, 1 = completely different)."""
    if a is None or b is None or a.shape != b.shape:
        return 1.0
    total = a.sum() + b.sum()
    return float(np.abs(a - b).sum() / total) if total > 0 else 0.0


def _covered(person_mask, bbox):
    x0, y0, x1, y1 = bbox
    region = person_mask[max(0, y0):y1, max(0, x0):x1]
    return float((region > 0).mean()) if region.size else 0.0
