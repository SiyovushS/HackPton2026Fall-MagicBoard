"""The exact corners of a hand-drawn box, using Darcy's box detection (virtual-cam-test.py):
Canny edges -> outer contours -> each simplified to 4 corners.

Qwen3-VL says WHICH box is which (its label, DEFINE or CALL) but its rectangle is rough; this
finds the real outline near it, so a saved drawing is cut from just inside the box's own lines and
a called drawing is warped into its four corners (tilted / perspective boxes too).

    quad = find_quad(board, rough_bbox)          # 4x2 float32: top-left, top-right, bottom-right, bottom-left, or None
    inside = rectify(board, quad, inset=0.06)    # the box's inside, straightened
    paste_into_quad(frame, drawing_bgra, quad)   # draw a saved drawing inside the box

    track = BoxTrack(quad)                       # follow a box from frame to frame (like Darcy's)
    quad = track.step(frame, person_mask)        # where it is now, or None once it's erased
"""
import cv2
import numpy as np


def find_quad(board, bbox, grow=0.25):
    """The drawn box near `bbox` (frame pixels) as 4 ordered corners, or None."""
    H, W = board.shape[:2]
    x0, y0, x1, y1 = bbox
    gw, gh = int((x1 - x0) * grow), int((y1 - y0) * grow)
    cx0, cy0, cx1, cy1 = max(0, x0 - gw), max(0, y0 - gh), min(W, x1 + gw), min(H, y1 + gh)
    crop = board[cy0:cy1, cx0:cx1]
    if crop.size == 0:
        return None
    best = None
    for quad in _quads_in(crop, min_area=max(64 ** 2 / 4, 0.3 * (x1 - x0) * (y1 - y0))):
        quad = (quad + (cx0, cy0)).astype(np.float32)
        score = _overlap(quad, bbox)
        if score > 0.5 and (best is None or score > best[0]):
            best = (score, quad)
    return best[1] if best else None


def find_all_quads(frame, work=960):
    """Every drawn box in the whole frame (Darcy's per-frame search), at a reduced size for speed."""
    s = min(1.0, work / max(frame.shape[:2]))
    small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else frame
    return [(q / s).astype(np.float32) for q in _quads_in(small, min_area=(64 * s) ** 2)]


def _quads_in(img, min_area):
    """Darcy's detection: Canny edges -> outer contours -> 4 corners each (ordered)."""
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blurred, 20, 80)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    # join an outline broken into pieces (small gaps in chalk lines; a label touching the side)
    k = max(5, int(0.03 * min(img.shape[:2]))) | 1
    joined = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    quads = []
    for c in contours:
        # Chalk outlines often have small gaps, so the outline isn't one closed shape: use the
        # outer hull of each edge piece, then check the 4 sides are really drawn.
        hull = cv2.convexHull(c)
        area = cv2.contourArea(hull)
        if len(hull) < 4 or area < min_area:
            continue
        approx = _four_corners(hull)
        if approx is None or cv2.contourArea(approx) < 0.85 * area:
            continue                                     # not box-shaped
        quad = approx.reshape(4, 2).astype(np.float32)
        if _sides_drawn(edges, quad) < 0.6:
            continue                                     # a blob's hull, not a drawn box
        quads.append(_order(quad))
    return quads


def rectify(img, quad, inset=0.06):
    """The inside of the quad warped to an upright rectangle (outline left out by `inset`)."""
    quad = np.asarray(quad, np.float32)
    w, h = _size(quad)
    if w < 8 or h < 8:
        return None
    M = cv2.getPerspectiveTransform(quad, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    flat = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_NEAREST if img.ndim == 2 else cv2.INTER_LINEAR)
    dx, dy = int(w * inset), int(h * inset)
    return flat[dy:h - dy, dx:w - dx]


def paste_into_quad(frame, art_bgra, quad, inset=0.12):
    """Draw a saved drawing (BGRA, transparent background) inside the quad, keeping proportions."""
    quad = np.asarray(quad, np.float32)
    w, h = _size(quad)
    if w < 8 or h < 8:
        return
    canvas = np.zeros((h, w, 4), np.uint8)               # the box, straightened
    bw, bh = int(w * (1 - 2 * inset)), int(h * (1 - 2 * inset))
    ah, aw = art_bgra.shape[:2]
    s = min(bw / aw, bh / ah)
    if s <= 0:
        return
    art = cv2.resize(art_bgra, (max(1, int(aw * s)), max(1, int(ah * s))), interpolation=cv2.INTER_AREA)
    ox, oy = (w - art.shape[1]) // 2, (h - art.shape[0]) // 2
    canvas[oy:oy + art.shape[0], ox:ox + art.shape[1]] = art
    # warp only into the quad's bounding rectangle (cheap enough for every frame)
    H, W = frame.shape[:2]
    rx0, ry0 = np.floor(quad.min(axis=0)).astype(int)
    rx1, ry1 = np.ceil(quad.max(axis=0)).astype(int)
    rx0, ry0, rx1, ry1 = max(0, rx0), max(0, ry0), min(W, rx1), min(H, ry1)
    if rx1 <= rx0 or ry1 <= ry0:
        return
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), (quad - (rx0, ry0)).astype(np.float32))
    warped = cv2.warpPerspective(canvas, M, (rx1 - rx0, ry1 - ry0))
    alpha = warped[..., 3:4].astype(np.float32) / 255
    region = frame[ry0:ry1, rx0:rx1].astype(np.float32)
    frame[ry0:ry1, rx0:rx1] = (alpha * warped[..., :3] + (1 - alpha) * region).astype(np.uint8)


def _four_corners(hull):
    """Simplify the outline more and more until 4 corners are left (None if it never is)."""
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.12, 23):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx
        if len(approx) < 4:
            return None
    return None


def _sides_drawn(edges, quad, step=4):
    """Share of the quad's outline that lies on (dilated) edges."""
    hits = total = 0
    for a, b in zip(quad, np.roll(quad, -1, axis=0)):
        n = max(2, int(np.linalg.norm(b - a) / step))
        for t in np.linspace(0, 1, n):
            x, y = (a + (b - a) * t).astype(int)
            if 0 <= y < edges.shape[0] and 0 <= x < edges.shape[1]:
                hits += edges[max(0, y - 3):y + 4, max(0, x - 3):x + 4].any()
            total += 1
    return hits / max(total, 1)


def _order(q):
    """Corners as top-left, top-right, bottom-right, bottom-left (Darcy's code kept the
    contour's own order, which can start at any corner and flip the picture)."""
    s, d = q.sum(axis=1), q[:, 0] - q[:, 1]
    return np.float32([q[np.argmin(s)], q[np.argmax(d)], q[np.argmax(s)], q[np.argmin(d)]])


def _size(q):
    w = int(max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3])))
    h = int(max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1])))
    return w, h


def _overlap(quad, bbox):
    """IoU of the quad's bounding rectangle with the rough box."""
    qx0, qy0 = quad.min(axis=0)
    qx1, qy1 = quad.max(axis=0)
    x0, y0, x1, y1 = bbox
    iw, ih = max(0, min(qx1, x1) - max(qx0, x0)), max(0, min(qy1, y1) - max(qy0, y0))
    inter = iw * ih
    return inter / ((qx1 - qx0) * (qy1 - qy0) + (x1 - x0) * (y1 - y0) - inter)


class BoxTrack:
    """Follows one box from frame to frame with Darcy's tracking (virtual-cam-test.py):
    each frame the box is found again (near where it was; if not there, anywhere in the frame,
    nearest within MAX_DIST) and the picture is pulled toward it by a damped spring -- his
    "wiggle":  vel *= 0.7;  vel += 0.1 * (found - pos);  pos += vel.
    His confidence count decides when the box is gone: +1 per frame it's found (up to 32),
    -1 per frame it isn't; below 0 -> step() returns None. While someone stands in front of
    the box the count isn't lowered (it's hidden, not erased)."""
    DAMPING, ACCEL = 0.7, 0.1      # Darcy's spring
    MAX_CONFIDENCE = 32
    MAX_DIST = 128 / 1280          # mean corner distance for a match, as a share of frame width (his 128 px)
    COVERED = 0.10                 # person over more than 10% of the box: don't count a miss

    def __init__(self, quad):
        self.pos = quad.astype(np.float32).copy()
        self.vel = np.zeros((4, 2), np.float32)    # one per box (Darcy's was shared by all)
        self.confidence = 8

    def step(self, frame, person=None):
        H, W = frame.shape[:2]
        x0, y0 = np.floor(self.pos.min(axis=0)).astype(int)
        x1, y1 = np.ceil(self.pos.max(axis=0)).astype(int)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        covered = x1 > x0 and y1 > y0 and person is not None and \
            (person[y0:y1, x0:x1] > 0).mean() > self.COVERED
        found = None
        if x1 > x0 and y1 > y0 and not covered:
            found = find_quad(frame, (x0, y0, x1, y1))
        if found is None and not covered:                  # moved further: search the whole frame
            found = self._nearest(find_all_quads(frame), W)
        self.vel *= self.DAMPING
        if found is not None:
            self.vel += (self.ACCEL * (found - self.pos)).astype(np.float32)
            self.confidence = min(self.confidence + 1, self.MAX_CONFIDENCE)
        elif not covered:
            self.confidence -= 1
        self.pos += self.vel
        return None if self.confidence < 0 else self.pos

    def _nearest(self, quads, width):
        size = np.linalg.norm(self.pos[2] - self.pos[0])
        best = None
        for q in quads:
            if not 0.6 < np.linalg.norm(q[2] - q[0]) / max(size, 1) < 1.6:
                continue                                   # a different-sized box
            d = np.linalg.norm(q - self.pos, axis=1).mean()
            if d < self.MAX_DIST * width and (best is None or d < best[0]):
                best = (d, q)
        return best[1] if best else None
