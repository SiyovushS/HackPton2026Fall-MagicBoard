"""Approach 1: only look at ink that is NEW since the session started (fixed camera).

    tracker = NewInkTracker()
    for each processed frame (a few per second):
        tracker.step(frame, person_mask, now)
        for region in tracker.ready(now):           # new writing, finished, not covered
            crop = tracker.crop(region)              # from the person-free board picture
            ... find/read what is in it (Qwen3-VL or snippet.py), then tracker.mark_done(region, result)

Why: lamps, clocks, screens, the board's frame, old smudges and ghost writing are on the board
from the start, so they are never considered, without any rule about them.

How:
  board picture  updated every frame, but only where no person is (so it shows the board, not them)
  baseline       the board picture at the start: everything already there is ignored
  new ink        ink in the board picture that is not in the baseline (slightly widened, so a
                 small camera shake doesn't count as new ink)
  regions        new ink grouped into areas; READY when unchanged for STABLE_SECONDS and not
                 covered by a person; done again only if its ink changes; dropped if erased
Robustness (learned on a recorded lecture):
  - the board picture is the per-pixel MEDIAN of the last ~3 s, so a person passing by is not
    burned in even where the person mask misses them (hair, edges of clothes)
  - only ink on the board surface counts (board mask from the start frame): a clock's moving
    hands and a TV screen next to the board changed constantly and looked like new ink
  - ink must persist over several updates before it counts, so faint writing that flickers in
    and out of detection doesn't trigger again and again
Limit: the camera must stay still. If it moves, call reset() to take a new baseline.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from ocr_engine import ink_mask

STABLE_SECONDS = 1.5
GROUP = 0.035          # new ink within ~3.5% of the frame belongs to one region
MIN_NEW_PIXELS = 80    # less new ink than this (in the work-size picture) is noise
CHANGED = 0.3          # the region's new ink grew/shrank by >30% (and >60 px) -> it changed
MEDIAN_FRAMES = 9      # board picture = median of the last 9 updates (~3 s at 3 per second)
PERSIST = 0.6          # a pixel is new ink once it was ink in >60% of recent updates
WORK_SIZE = 1280


@dataclass
class InkRegion:
    id: int
    bbox: tuple            # frame pixels (x0, y0, x1, y1)
    pixels: int            # amount of new ink
    stable_since: float
    last_seen: float
    covered: bool = False
    done_pixels: int = -1  # ink amount when it was last processed (-1 = never)
    result: object = None


class NewInkTracker:
    def __init__(self, all_new=False):
        self.all_new = all_new     # True: nothing counts as already on the board (N key)
        self.reset()

    def reset(self):
        self.board = None          # person-free picture of the board (work size)
        self.recent = []           # last MEDIAN_FRAMES work-size frames
        self.on_board = None       # board surface (from the start frame)
        self.persist = None        # running share of updates in which each pixel was new ink
        self.baseline_ink = None   # ink already on the board at the start
        self.regions = []
        self._next = 1
        self.scale = 1.0

    def step(self, frame_bgr, person_mask, now):
        h0, w0 = frame_bgr.shape[:2]
        self.scale = min(1.0, WORK_SIZE / max(h0, w0))
        small = cv2.resize(frame_bgr, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        person = cv2.resize(person_mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        if self.board is None:
            if person.mean() > 0.01:                 # wait for a moment with nobody in front
                return
            from snippet import board_mask
            self.board = small.copy()
            self.board_kind = "white" if small.mean() > 110 else "black"
            self.baseline_ink = cv2.dilate(ink_mask(small, board=self.board_kind), np.ones((7, 7), np.uint8))
            if self.all_new:
                self.baseline_ink[:] = 0
            surface = board_mask(small)
            self.on_board = cv2.erode(surface, np.ones((15, 15), np.uint8)) > 0
            self.persist = np.zeros(small.shape[:2], np.float32)
            return
        self.recent = (self.recent + [small])[-MEDIAN_FRAMES:]
        median = np.median(np.stack(self.recent), axis=0).astype(np.uint8)
        person = cv2.dilate(person.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
        self.board[~person] = median[~person]       # update the board only where nobody stands
        new = ink_mask(self.board, board=self.board_kind)
        new[self.baseline_ink > 0] = 0
        new[~self.on_board] = 0                      # clock, screen, wall: not the board
        self.persist = 0.8 * self.persist + 0.2 * (new > 0)
        new = ((self.persist > PERSIST) * 255).astype(np.uint8)
        new = _drop_specks(new, 8)
        h, w = new.shape
        k = max(5, int(min(h, w) * GROUP) | 1)
        grouped = cv2.dilate(new, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, 8)
        found = []
        for i in range(1, n):
            x, y, bw, bh, _ = stats[i]
            px = int((new[labels == i] > 0).sum())
            if px < MIN_NEW_PIXELS:
                continue
            bbox = tuple(int(v / self.scale) for v in (x, y, x + bw, y + bh))
            covered = person[y:y + bh, x:x + bw].mean() > 0.05
            found.append((bbox, px, covered))
        seen = set()
        for bbox, px, covered in found:
            r = max((r for r in self.regions if r.id not in seen), key=lambda r: _iou(r.bbox, bbox), default=None)
            if r is None or _iou(r.bbox, bbox) < 0.2:
                r = InkRegion(self._next, bbox, px, now, now)
                self._next += 1
                self.regions.append(r)
            else:
                if abs(px - r.pixels) > max(60, CHANGED * r.pixels) or _iou(r.bbox, bbox) < 0.6:
                    r.stable_since = now                       # still being written
                r.bbox, r.pixels, r.last_seen = bbox, px, now
            r.covered = covered
            seen.add(r.id)
        self.regions = [r for r in self.regions if now - r.last_seen < 2.0]   # erased -> gone

    def ready(self, now):
        return [r for r in self.regions
                if not r.covered and now - r.stable_since >= STABLE_SECONDS
                and (r.done_pixels < 0 or abs(r.pixels - r.done_pixels) > max(60, CHANGED * r.done_pixels))]

    def crop(self, region, margin=0.25):
        """The region from the person-free board picture (full-res coordinates are approximate)."""
        x0, y0, x1, y1 = (int(v * self.scale) for v in region.bbox)
        m = int(max(x1 - x0, y1 - y0) * margin)
        h, w = self.board.shape[:2]
        x0, y0, x1, y1 = max(0, x0 - m), max(0, y0 - m), min(w, x1 + m), min(h, y1 + m)
        crop = self.board[y0:y1, x0:x1]
        return cv2.resize(crop, None, fx=1 / self.scale, fy=1 / self.scale, interpolation=cv2.INTER_CUBIC), \
            tuple(int(v / self.scale) for v in (x0, y0, x1, y1))

    def mark_done(self, region, result):
        region.done_pixels, region.result = region.pixels, result

    def new_ink_picture(self):
        """Debug: the board with new ink in red."""
        if self.board is None:
            return None
        out = self.board.copy()
        out[self.persist > PERSIST] = (0, 0, 255)
        out[~self.on_board] = (0.5 * out[~self.on_board]).astype(np.uint8)
        return out


def _drop_specks(mask, min_area):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return (keep[labels] * 255).astype(np.uint8)


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
