"""Snippeting: find the regions on the board that the app reads.

    boxes, axes, writing = snip_board(frame)          # everything, ~0.1-0.3 s per frame
    for s in writing:
        s.crop, s.hint, s.role   # role: "autocomplete" ("= ___"), "graph" (next to axes), "writing"
        # read it with:  engine.recognize(s.crop, hint=s.hint)

Free writing: ink with boxes and axes removed; letters joined sideways into lines (1.5 letter
heights) but barely up/down (0.5), then pieces of one expression merged (same-line neighbours,
overlapping parts, letter-sized limits/superscripts). Only regions whose background looks like
the board are kept, which drops lamps, walls, chairs and the board's frame.
Graph axes: one long column and one long row of ink meeting at a corner, empty between.

BOXES, for artifacts:
    boxes = detect_boxes(frame)
    for b in boxes:
        b.corners        # 4 corners in frame pixels: top-left, top-right, bottom-right, bottom-left
        b.label_crop     # bottom-left corner inside the box, where the label is written
        b.content_crop   # the inside of the box (straightened), minus the label corner
        b.has_content    # True: box with a drawing -> DEFINE artifact; False: empty -> CALL artifact

How boxes are found:
  A box is the only thing on a board that ENCLOSES a large empty area. The ink is thickened
  just enough to bridge small gaps in hand-drawn lines; each box then becomes a closed ring
  with a large hole inside, while text becomes solid blobs with only tiny holes (inside an
  "o"). Large, rectangular holes are the boxes; their 4 corners come from cv2.approxPolyN,
  the 4-corner simplification Darcy's quad detection (virtual-cam-test.py) used.
  Detection runs on both dark ink (marker) and light ink (chalk) separately, so one can't
  hide the other (a dark ledge edge once hid a chalk box when only one was used).
  Tested approaches that did NOT work and were removed: closed outlines only (hand-drawn
  boxes have gaps) and assembling boxes from straight line pieces (found boxes in text).
  Limit: lines that are missing from the image (e.g. washed out by glare) can't be bridged.

Run on an image to see what it finds:
    .venv/bin/python snippet.py board.png            # writes board_boxes.png next to it
"""
from dataclasses import dataclass

import cv2
import numpy as np

from ocr_engine import BOARD, find_answer_blank, has_ink, ink_mask, remove_edge_lines

WORK_SIZE = 1280         # frames are analysed at this size (longest side) for speed
MIN_AREA = 0.01          # a box covers at least 1% of the frame...
MAX_AREA = 0.97          # ...and not (almost) all of it
BRIDGE = 0.025           # ink is thickened to bridge gaps up to ~2.5% of the frame
MIN_RECTANGULAR = 0.85   # a box's hole fills at least 85% of its 4-corner outline
MIN_SIDE_DRAWN = 0.75    # open boxes: every side at least 75% drawn...
MEAN_SIDE_DRAWN = 0.85   # ...and 85% on average (measured on tests: boxes 81-98%)
LABEL_W, LABEL_H = 0.55, 0.32   # label corner: bottom-left 55% x 32% of the box's inside


@dataclass
class Box:
    corners: np.ndarray      # (4, 2) float, frame pixels: TL, TR, BR, BL
    inside: np.ndarray       # the box's inside, straightened, border excluded (BGR)
    label_crop: np.ndarray   # bottom-left corner of the inside (BGR)
    content_crop: np.ndarray # the inside with the label corner blanked out (BGR)
    has_content: bool        # drawing/writing inside besides the label -> define; else call
    rectangular: float       # how well the inside fits a 4-corner shape (0..1)

    @property
    def bbox(self):
        x0, y0 = self.corners.min(axis=0)
        x1, y1 = self.corners.max(axis=0)
        return int(x0), int(y0), int(x1), int(y1)


def detect_boxes(frame_bgr, debug=False):
    """Find hand-drawn boxes in a frame. Returns a list of Box, largest first."""
    h0, w0 = frame_bgr.shape[:2]
    scale = min(1.0, WORK_SIZE / max(h0, w0))
    small = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame_bgr
    h, w = small.shape[:2]
    boards = ("white", "black") if BOARD == "auto" else (BOARD,)
    found = []
    for board in boards:
        ink = _drop_specks(ink_mask(small, board=board), min_area=max(15, int(h * w * 2e-5)))
        for quad, rect in _box_holes(ink):
            if debug:
                print(f"  {board} ink: hole {cv2.contourArea(quad) / (h * w):.1%} of frame, rectangular={rect:.2f}")
            found.append((quad, rect + 1.0))   # +1: where both methods find a box, prefer the hole
                                               # (it hugs the inside, so no border leaks into the content)
        for quad, rect, sides in _open_boxes(ink):
            if debug:
                print(f"  {board} ink: open box {cv2.contourArea(quad) / (h * w):.1%} of frame, "
                      f"rectangular={rect:.2f}, sides drawn={[f'{x:.0%}' for x in sides]}")
            found.append((quad, rect))
    found = _remove_duplicates(found, (h, w))
    boxes = []
    for quad, score in found:
        rect = score - 1.0 if score > 1.0 else score   # undo the +1 preference given to holes
        box = _make_box(frame_bgr, quad / scale, rect)
        if box is not None:
            boxes.append(box)
    boxes.sort(key=lambda b: -cv2.contourArea(b.corners))
    return boxes


def _box_holes(ink):
    """Large rectangular holes enclosed by (gap-bridged) ink: (corners, rectangularity)."""
    h, w = ink.shape
    k = max(5, int(min(h, w) * BRIDGE) | 1)
    ring = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    contours, hierarchy = cv2.findContours(ring, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c, (_, _, _, parent) in zip(contours, [] if hierarchy is None else hierarchy[0]):
        if parent < 0:                                    # outer outline, not a hole
            continue
        area = cv2.contourArea(c)
        if not (MIN_AREA * h * w <= area <= MAX_AREA * h * w):
            continue
        try:
            quad = cv2.approxPolyN(cv2.convexHull(c), nsides=4).reshape(-1, 2).astype(np.float32)
        except cv2.error:
            continue
        rect = area / max(cv2.contourArea(quad), 1)
        if len(quad) != 4 or rect < MIN_RECTANGULAR:
            continue
        # the hole's edge runs along the inside of the thickened line: move corners out to the line
        centre = quad.mean(axis=0)
        quad = quad + (quad - centre) / np.linalg.norm(quad - centre, axis=1, keepdims=True) * (k / 2)
        out.append((_order_corners(quad), rect))
    return out


def _open_boxes(ink):
    """Boxes whose outline is not quite closed (e.g. one open corner), so they have no hole.

    For each large connected stroke: simplify its outer outline to 4 corners and accept it if
    the stroke runs along most of all 4 sides and the inside is mostly empty. Text blobs fail
    (not hollow), drawings like an apple fail (no straight sides), axes fail (only 2 sides).
    Returns (corners, rectangularity, per-side coverage).
    """
    h, w = ink.shape
    k = max(5, int(min(h, w) * BRIDGE) | 1)
    thick = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    kk = max(5, int(k * 2.5) | 1)   # hand-drawn sides bow: allow some distance from a straight side
    near = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))) > 0
    ink_k = thick > 0                # tighter tolerance, for the lines just inside the edges
    letter = _letter_height(ink) or 0
    n, labels, _, _ = cv2.connectedComponentsWithStats(thick, 8)
    out = []
    for i in range(1, n):
        comp = labels == i
        c = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if not (MIN_AREA * h * w <= hull_area <= MAX_AREA * h * w):
            continue
        try:
            quad = cv2.approxPolyN(hull, nsides=4).reshape(-1, 2).astype(np.float32)
        except cv2.error:
            continue
        rect = hull_area / max(cv2.contourArea(quad), 1)
        if len(quad) != 4 or rect < MIN_RECTANGULAR:
            continue
        # the outline runs along the outside of the thickened line: move corners in to the line
        centre = quad.mean(axis=0)
        quad = quad - (quad - centre) / np.linalg.norm(quad - centre, axis=1, keepdims=True) * (k / 2)
        quad = _order_corners(quad)
        sides = _side_coverage(quad, near)
        # A box OUTLINE is hollow; a text blob is dense. Count only this stroke's own ink in the
        # middle of the box (sides excluded): a separate drawing inside (a DEFINE box's content)
        # is a different stroke and must not count, or every box with a drawing is rejected.
        middle = _poly_mask(quad.mean(axis=0) + (quad - quad.mean(axis=0)) * 0.7, (h, w)) > 0
        inside_ink = ((ink > 0) & comp & middle).sum() / max(1, middle.sum())
        # A box has ink ON its edges but not just inside them. Lines parallel to the sides, 20% in,
        # cross only whatever is drawn inside: measured 0-1% for empty boxes, 28% for a box with a
        # house drawing, 43-80% for blocks of text (which also reach all 4 edges) and a lamp.
        centre = quad.mean(axis=0)
        inner = np.mean(_side_coverage(centre + (quad - centre) * 0.8, ink_k))
        tall_enough = min(np.ptp(quad[:, 0]), np.ptp(quad[:, 1])) >= 3 * letter   # not a letter in a crop
        if inner > 0.35 or not tall_enough or inside_ink > 0.08:
            continue
        well_drawn = min(sides) >= MIN_SIDE_DRAWN and np.mean(sides) >= MEAN_SIDE_DRAWN
        # big, with one side partly missing (faint, or washed out by glare)
        big_partial = hull_area >= 0.05 * h * w and np.mean(sides) >= 0.75
        if well_drawn or big_partial:
            out.append((quad, rect, sides))
    return out


def _side_coverage(quad, near_ink, samples=80):
    """Share of points along each of the 4 sides that lie on (thickened) ink."""
    h, w = near_ink.shape
    out = []
    for a, b in zip(quad, np.roll(quad, -1, axis=0)):
        pts = a + np.linspace(0, 1, samples)[:, None] * (b - a)
        xs = np.clip(pts[:, 0].round().astype(int), 0, w - 1)
        ys = np.clip(pts[:, 1].round().astype(int), 0, h - 1)
        out.append(float(near_ink[ys, xs].mean()))
    return out


# ======================= step 2: free writing and graph axes =======================
@dataclass
class Snippet:
    bbox: tuple              # (x0, y0, x1, y1) in frame pixels
    crop: np.ndarray         # the region cut from the frame (BGR), with a small margin
    hint: object             # "math" (answer blank / next to axes) or None (auto) for the OCR engine
    role: str                # "writing", "autocomplete" (has "= ___"), or "graph" (next to axes)


@dataclass
class Axes:
    bbox: tuple              # (x0, y0, x1, y1) in frame pixels: the drawn graph axes
    corner: tuple            # where the two axes meet (x, y): the graph's origin


def snip_board(frame_bgr, debug=False, ignore=None):
    """Everything to read on the board: (boxes, axes, writing snippets).

    ignore: optional mask (e.g. people.PersonFinder().mask(frame)); anything under it is treated
    as blank board, so a person's hair, arm or clothes are never read as writing.
    """
    off_board = board_mask(frame_bgr) == 0       # chairs, lamp, wall, frame: never writing
    ignore = off_board if ignore is None else (ignore > 0) | off_board
    if ignore.any():
        frame_bgr = _blank_out(frame_bgr, ignore.astype(np.uint8))
    boxes = detect_boxes(frame_bgr, debug=debug)
    axes = detect_axes(frame_bgr, boxes)
    writing = detect_writing(frame_bgr, boxes, axes, debug=debug)
    return boxes, axes, writing


def board_mask(frame_bgr):
    """The board surface: the large connected area of board colour around the frame's middle,
    with the writing on it included. Everything else (chairs in front, lamp, wall, the board's
    frame) is 0. Board colour = median of the frame's middle, as in _on_board."""
    small, scale = _work_copy(frame_bgr)
    h, w = small.shape[:2]
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    board = np.median(lab[h // 4: 3 * h // 4, w // 4: 3 * w // 4].reshape(-1, 3), axis=0)
    like_board = ((np.abs(lab[..., 0] - board[0]) <= BOARD_BRIGHTNESS_DIFF) &
                  (np.hypot(lab[..., 1] - board[1], lab[..., 2] - board[2]) <= BOARD_HUE_DIFF)).astype(np.uint8)
    k = max(5, int(min(h, w) * 0.03) | 1)                    # absorb the writing and small smudges
    like_board = cv2.morphologyEx(like_board, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(like_board, 8)
    if n <= 1:
        return np.full(frame_bgr.shape[:2], 255, np.uint8)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == biggest).astype(np.uint8) * 255
    filled = mask.copy()                                      # fill holes: big drawings, boxes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, contours, -1, 255, -1)
    return cv2.resize(filled, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)


def _blank_out(frame_bgr, mask):
    """Paint the masked area in the board's colour (median of the rest of the frame's middle)."""
    h, w = mask.shape
    middle = frame_bgr[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    free = mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4] == 0
    colour = np.median(middle[free], axis=0) if free.any() else np.median(frame_bgr.reshape(-1, 3), axis=0)
    out = frame_bgr.copy()
    out[mask > 0] = colour.astype(np.uint8)
    return out


def detect_axes(frame_bgr, boxes=()):
    """Graph axes: an L of two long, straight, roughly perpendicular lines with empty space
    between them (the graph trigger). Found like open boxes, but with only 2 adjacent sides."""
    small, scale = _work_copy(frame_bgr)
    h, w = small.shape[:2]
    ink = _drop_specks(ink_mask(small), max(15, int(h * w * 2e-5)))
    ink[_boxes_mask(boxes, (h, w), scale) > 0] = 0
    k = max(5, int(min(h, w) * BRIDGE) | 1)
    thick = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))  # tolerates slight tilt
    letter = _letter_height(ink) or 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(thick, 8)
    found = []
    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        # both arms long: relative to the frame, and at least 3 letters (else an "L" or "=" in a crop counts)
        if bw < max(0.08 * w, 3 * letter) or bh < max(0.08 * h, 3 * letter):
            continue
        comp = labels[y:y + bh, x:x + bw] == i
        cols, rows = comp.sum(axis=0), comp.sum(axis=1)
        cx, cy = int(np.argmax(cols)), int(np.argmax(rows))         # the vertical and horizontal axis
        # each axis runs (almost) the full height / width; arrowheads don't matter
        if cols[cx] < 0.75 * bh or rows[cy] < 0.75 * bw:
            continue
        near_left, near_right = cx < 0.25 * bw, cx > 0.75 * bw
        near_top, near_bottom = cy < 0.25 * bh, cy > 0.75 * bh
        if not ((near_left or near_right) and (near_top or near_bottom)):   # they meet at a corner
            continue
        # empty between the arms: no ink away from the two axis lines
        between = comp.copy()
        between[:, max(0, cx - k):cx + k + 1] = False
        between[max(0, cy - k):cy + k + 1, :] = False
        if between.sum() > 0.08 * comp.sum():                       # allow arrowheads, nothing more
            continue
        found.append(Axes(tuple(int(v / scale) for v in (x, y, x + bw, y + bh)),
                          (float((x + cx) / scale), float((y + cy) / scale))))
    return found


def detect_writing(frame_bgr, boxes=(), axes=(), debug=False):
    """Free writing outside boxes, grouped into snippets (one per expression / line of text).

    Letters are joined into lines by thickening the ink much more sideways (1.5x the typical
    letter height) than up/down (0.5x), so words merge into lines but separate lines stay apart,
    while superscripts and fraction parts still join their expression.
    """
    small, scale = _work_copy(frame_bgr)
    h, w = small.shape[:2]
    ink = _drop_specks(ink_mask(small), max(15, int(h * w * 2e-5)))
    ink[_boxes_mask(boxes, (h, w), scale) > 0] = 0               # boxes are handled separately
    for a in axes:                                                # so are graph axes
        x0, y0, x1, y1 = (int(v * scale) for v in a.bbox)
        ink[max(0, y0 - 3):y1 + 4, max(0, x0 - 3):x1 + 4] = 0
    ink = _drop_long_lines(ink)                                   # board edges, panel seams, trays
    letter = _letter_height(ink)
    if letter is None:
        return []
    joined = cv2.dilate(ink, np.ones((max(3, int(letter * 0.5)), max(3, int(letter * 1.5))), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    groups = [tuple(stats[i][:4]) for i in range(1, n)
              if stats[i][3] >= letter * 0.6 and stats[i][2] >= letter * 0.6]       # drop specks and dots
    groups = _merge_overlapping(groups, pad=int(letter * 1.0))   # rejoin split expressions (x^2, integrals)
    board_lab = _board_colour(small, ink)
    if debug:
        print(f"  writing: letter height {letter:.0f}px, {len(groups)} groups, board colour (Lab) {np.round(board_lab)}")
    snippets = []
    margin = max(4, int(letter * 0.4))
    for x, y, bw, bh in groups:
        if not _enough_writing(ink[y:y + bh, x:x + bw], bw, letter):
            continue                                              # smudges, ticks, dots
        if not _on_board(small[y:y + bh, x:x + bw], ink[y:y + bh, x:x + bw], board_lab):
            continue                                              # lamp, wall, chairs, board frame
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(w, x + bw + margin), min(h, y + bh + margin)
        box = tuple(int(v / scale) for v in (x0, y0, x1, y1))
        crop = frame_bgr[box[1]:box[3], box[0]:box[2]].copy()
        role, hint = "writing", None
        if find_answer_blank(remove_edge_lines(crop)) is not None:
            role, hint = "autocomplete", "math"
        elif any(_left_of_axes(box, a) for a in axes):
            role, hint = "graph", "math"
        snippets.append(Snippet(box, crop, hint, role))
    snippets.sort(key=lambda s: (s.bbox[1] // max(1, int(letter / scale)), s.bbox[0]))   # reading order
    return snippets


def _drop_long_lines(ink, span=0.3):
    """Remove long thin strokes spanning over 30% of the frame: the board's frame edges, panel
    seams and trays. No handwriting stroke is that long, and left in they join everything on
    the board into one huge snippet. (Boxes are found separately, before this.)"""
    h, w = ink.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    out = ink.copy()
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        long_line = bw >= span * w or bh >= span * h
        thin = area <= 0.15 * bw * bh                      # a line, not a block of writing
        if long_line and thin:
            out[labels == i] = 0
    return out


MIN_INK_LETTERS = 0.6      # a snippet needs at least this much ink (in letter-sized squares)...
MIN_WIDTH_LETTERS = 4.0    # ...and this width (writing only, no margin), unless it has lots of ink
                           # ("2+2" is ~4.4 letters wide; 5.0 dropped it)
LOTS_OF_INK_LETTERS = 2.5  # (e.g. a tall, narrow integral sign)


def _enough_writing(ink_region, width, letter):
    """Is there enough here to be writing? Measured on tests/board: all 23 real snippets had
    >= 0.62 letters of ink and were >= 5 letters wide (bar one tall integral sign, 2.6 letters of
    ink); 17 of 32 junk snippets (smudges, ticks, dots) fell below. A lone single character is
    skipped too, which is the trade-off."""
    ink_letters = stroke_pixels_in(ink_region) / (letter * letter)
    if ink_letters >= LOTS_OF_INK_LETTERS:
        return True
    return ink_letters >= MIN_INK_LETTERS and width >= MIN_WIDTH_LETTERS * letter


def stroke_pixels_in(mask):
    return int((mask > 0).sum())


# A snippet's background must look like the board. Brightness may differ more (glare, shadows:
# a glare-lit "x^2+3=___" was 43 brighter) than hue (chairs, wall: 14-50 apart; board: <= 3).
BOARD_BRIGHTNESS_DIFF = 50
BOARD_HUE_DIFF = 12


def _board_colour(small, ink):
    """The board's colour: median of non-ink pixels in the middle half of the frame (where the
    camera points). Lab colour, so brightness and hue differences are both measured."""
    h, w = ink.shape
    region = small[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    blank = ink[h // 4: 3 * h // 4, w // 4: 3 * w // 4] == 0
    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB).reshape(-1, 3)[blank.ravel()]
    return np.median(lab, axis=0) if len(lab) else np.array([128, 128, 128])


def _on_board(region_bgr, region_ink, board_lab):
    """Is this snippet written on the board? Its background (non-ink pixels) must look like the
    board. Rejects things that are not the board: lamps, walls, chairs, the board's frame."""
    lab = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3)[(region_ink == 0).ravel()]
    if len(lab) == 0:
        return False
    m = np.median(lab, axis=0).astype(float)
    brightness = abs(m[0] - board_lab[0])
    hue = float(np.hypot(m[1] - board_lab[1], m[2] - board_lab[2]))
    return brightness <= BOARD_BRIGHTNESS_DIFF and hue <= BOARD_HUE_DIFF


def _same_line(a, b, gap):
    """Side by side on the same line, close together (an integral sign and its body)."""
    ix = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    iy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return iy >= 0.5 * min(a[3], b[3]) and -ix <= gap


def _mostly_overlap(a, b, gap=0):
    """One piece mostly lies on the other (a split-off superscript or fraction part)."""
    ix = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    iy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return ix > 0 and iy > 0 and ix * iy >= 0.5 * min(a[2] * a[3], b[2] * b[3])


def _small_touching(a, b, gap):
    """A letter-sized piece (an integral limit, superscript, dot) touching a bigger one.
    `gap` is one letter height. "Letter-sized" matters: a short expression like "x = <-o->"
    next to a paragraph is smaller than the paragraph but is not part of it."""
    letter = gap
    pad = letter * 0.3
    touching = (a[0] - pad < b[0] + b[2] and b[0] - pad < a[0] + a[2] and
                a[1] - pad < b[1] + b[3] and b[1] - pad < a[1] + a[3])
    small, big = sorted((a, b), key=lambda r: r[2] * r[3])
    letter_sized = max(small[2], small[3]) <= 2.5 * letter   # limits are often written large
    return touching and letter_sized and small[2] * small[3] <= 0.25 * big[2] * big[3]


def _merge_overlapping(rects, pad=0):
    """Merge pieces (x, y, w, h) of the same expression. Same-line neighbours first, so a piece
    like the "ax" of "ax^2+bx+c" joins its own line before it could be caught by the line below;
    then pieces that mostly overlap; then small pieces (limits, superscripts) touching a big one.
    Lines stacked above each other stay separate."""
    for rule in (_same_line, _mostly_overlap, _small_touching):
        rects = _merge_by(rects, rule, pad)
    return rects


def _merge_by(rects, rule, pad):
    rects = [list(r) for r in rects]
    merged = True
    while merged:
        merged = False
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                a, b = rects[i], rects[j]
                if rule(a, b, pad):
                    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
                    rects[i] = [x0, y0, x1 - x0, y1 - y0]
                    del rects[j]
                    merged = True
                    break
            if merged:
                break
    return [tuple(r) for r in rects]


def _left_of_axes(box, axes):
    """An expression written just left of (and level with) the axes is the one to graph."""
    x0, y0, x1, y1 = box
    ax0, ay0, ax1, ay1 = axes.bbox
    level = min(y1, ay1) - max(y0, ay0) > 0.3 * (y1 - y0)
    gap = ax0 - x1                     # may be slightly negative: snippet margins can overlap the axes
    return level and -0.15 * (ax1 - ax0) <= gap <= (ax1 - ax0) * 0.8


def _letter_height(ink):
    """Typical letter height: median height of letter-sized ink blobs."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    h = ink.shape[0]
    heights = [s[cv2.CC_STAT_HEIGHT] for s in stats[1:] if 0.008 * h <= s[cv2.CC_STAT_HEIGHT] <= 0.15 * h]
    return float(np.median(heights)) if len(heights) >= 3 else None


def _work_copy(frame_bgr):
    h0, w0 = frame_bgr.shape[:2]
    scale = min(1.0, WORK_SIZE / max(h0, w0))
    small = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame_bgr
    return small, scale


def _boxes_mask(boxes, shape, scale):
    """Mask covering each box, slightly enlarged so its border line is included."""
    m = np.zeros(shape, np.uint8)
    for b in boxes:
        q = b.corners * scale
        centre = q.mean(axis=0)
        q = centre + (q - centre) * 1.08
        cv2.fillConvexPoly(m, q.astype(np.int32), 255)
    return m


def draw_snippets(frame_bgr, boxes, axes, writing, readings=None):
    """Debug picture: boxes (green/blue), axes (magenta), writing snippets (orange/red/purple)."""
    out = draw_boxes(frame_bgr, boxes)
    t = max(2, out.shape[1] // 600)
    for a in axes:
        cv2.rectangle(out, a.bbox[:2], a.bbox[2:], (255, 0, 255), t)
        cv2.putText(out, "axes (graph)", (a.bbox[0], a.bbox[1] - 4 * t), cv2.FONT_HERSHEY_SIMPLEX, t * 0.4, (255, 0, 255), t)
    colours = {"writing": (0, 140, 255), "autocomplete": (0, 0, 230), "graph": (200, 0, 200)}
    for i, s in enumerate(writing):
        c = colours[s.role]
        cv2.rectangle(out, s.bbox[:2], s.bbox[2:], c, t)
        text = f"{i + 1}: {s.role}"
        if readings and i < len(readings) and readings[i]:
            text += f"  {readings[i]}"
        cv2.putText(out, text[:60], (s.bbox[0], s.bbox[1] - 3 * t), cv2.FONT_HERSHEY_SIMPLEX, t * 0.4, c, t)
    return out


def draw_boxes(frame_bgr, boxes):
    """Debug picture: each box outlined, its label corner marked, and DEFINE/CALL written on it."""
    out = frame_bgr.copy()
    t = max(2, out.shape[1] // 500)
    for i, b in enumerate(boxes):
        colour = (0, 170, 0) if b.has_content else (255, 120, 0)
        cv2.polylines(out, [b.corners.astype(np.int32)], True, colour, t * 2)
        tl, tr, br, bl = b.corners
        lab = np.array([bl, bl + (br - bl) * LABEL_W, bl + (br - bl) * LABEL_W + (tl - bl) * LABEL_H,
                        bl + (tl - bl) * LABEL_H], np.int32)
        cv2.polylines(out, [lab], True, (0, 200, 255), t)
        text = f"box {i + 1}: {'DEFINE (has content)' if b.has_content else 'CALL (empty)'}"
        cv2.putText(out, text, (int(tl[0]), int(tl[1]) - 4 * t), cv2.FONT_HERSHEY_SIMPLEX, t * 0.45, colour, t)
    return out


# ---------- helpers ----------
def _drop_specks(mask, min_area):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return (keep[labels] * 255).astype(np.uint8)


def _order_corners(quad):
    s, d = quad.sum(axis=1), np.diff(quad, axis=1).ravel()
    return np.array([quad[np.argmin(s)], quad[np.argmin(d)], quad[np.argmax(s)], quad[np.argmax(d)]], np.float32)


def _poly_mask(quad, shape):
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, quad.astype(np.int32), 255)
    return m


def _remove_duplicates(found, shape):
    """The same box can be found in both ink directions: keep one where boxes overlap."""
    keep = []
    for quad, score in sorted(found, key=lambda qs: (-qs[1], -cv2.contourArea(qs[0]))):
        m = _poly_mask(quad, shape) > 0
        if all((m & k).sum() < 0.5 * min(m.sum(), k.sum()) for _, _, k in keep):
            keep.append((quad, score, m))
    return [(q, s) for q, s, _ in keep]


def _make_box(frame_bgr, corners, rectangular):
    tl, tr, br, bl = corners
    bw = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    bh = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    m = cv2.getPerspectiveTransform(corners.astype(np.float32),
                                    np.float32([[0, 0], [bw - 1, 0], [bw - 1, bh - 1], [0, bh - 1]]))
    straight = cv2.warpPerspective(frame_bgr, m, (bw, bh), borderMode=cv2.BORDER_REPLICATE)
    inset = max(3, int(min(bw, bh) * 0.07))                    # skip the (wobbly) border line itself
    inside = straight[inset:bh - inset, inset:bw - inset]
    ih, iw = inside.shape[:2]
    if ih < 8 or iw < 8:
        return None   # degenerate (tiny or squashed) quad
    label = inside[int(ih * (1 - LABEL_H)):, :int(iw * LABEL_W)].copy()
    content = inside.copy()
    content[int(ih * (1 - LABEL_H)):, :int(iw * LABEL_W)] = cv2.medianBlur(inside, 21)[int(ih * (1 - LABEL_H)):, :int(iw * LABEL_W)]
    return Box(corners.astype(np.float32), inside, label, content, has_ink(content), rectangular)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    for path in sys.argv[1:]:
        img = cv2.imread(path)
        if img is None:
            print(f"{path}: could not read")
            continue
        found, axes, writing = snip_board(img, debug=True)
        print(f"{path}: {len(found)} box(es), {len(axes)} axes, {len(writing)} writing snippet(s)")
        for i, b in enumerate(found):
            print(f"  box {i + 1}: bbox={b.bbox} rectangular={b.rectangular:.2f} "
                  f"-> {'DEFINE (has content)' if b.has_content else 'CALL (empty)'}")
        for a in axes:
            print(f"  axes: bbox={a.bbox} corner={tuple(round(v) for v in a.corner)}")
        for i, s in enumerate(writing):
            print(f"  snippet {i + 1}: bbox={s.bbox} role={s.role} hint={s.hint}")
        out = Path(path).with_name(Path(path).stem + "_boxes.png")
        cv2.imwrite(str(out), draw_snippets(img, found, axes, writing))
        print(f"  picture: {out}")
