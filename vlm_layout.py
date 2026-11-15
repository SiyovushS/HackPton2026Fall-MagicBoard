"""Approach 2: let the vision model find the regions on the board (instead of snippet.py's rules).

    boxes, axes, writing = find_with_vlm(engine, frame)    # same shapes as snippet.snip_board

Qwen3-VL gets the whole board and lists every box (with its label and whether something is
drawn inside), expression, line of text and graph axes, with coordinates. It understands that
a smudge, a chair or the board's frame is not writing, which pixel rules can't. Each item also
carries the model's own reading (`.text`), so it can be used directly, or each region can be
cropped and read again with OCREngine.recognize for crop-level accuracy.
"""
import json
import re
from dataclasses import dataclass

import numpy as np

from snippet import LABEL_H, LABEL_W, _left_of_axes

PROMPT = """This is a photo of a lecture board (a whiteboard or a chalkboard).
List every item written or drawn ON THE BOARD as a JSON array. Each item is an object:
  {"type": "box", "bbox_2d": [x1, y1, x2, y2], "label": "<the small word written inside the box, near its bottom-left corner, or null>", "has_drawing": <true if anything besides the label is drawn inside the box>}
  {"type": "expression", "bbox_2d": [...], "latex": "<the math, as LaTeX>", "blank": <true if it ends with an empty answer line like "= ___">}
  {"type": "text", "bbox_2d": [...], "text": "<the words>"}
  {"type": "axes", "bbox_2d": [...]}            (hand-drawn graph axes: an L shape with arrows)
A "box" is a hand-drawn rectangle; do not also list its label or drawing as separate items.
Give one item per expression or line of text. Ignore smudges, eraser marks, the board's frame,
furniture, lamps and people. Output only the JSON array."""


@dataclass
class VBox:
    bbox: tuple
    corners: np.ndarray
    label_crop: np.ndarray
    has_content: bool
    text: str            # the model's reading of the label


@dataclass
class VAxes:
    bbox: tuple


@dataclass
class VSnippet:
    bbox: tuple
    crop: np.ndarray
    hint: object
    role: str
    text: str            # the model's own reading


def find_with_vlm(engine, frame_bgr, max_tokens=1200, debug=False):
    h, w = frame_bgr.shape[:2]
    raw = engine._ask_many([frame_bgr], [PROMPT], max_tokens)[0]
    items = _parse_json(raw)
    if debug:
        print(raw[:2000])
    # Qwen3-VL gives coordinates on a 0-1000 scale; older Qwen-VL versions used pixels
    coords = [v for it in items for v in (it.get("bbox_2d") or [])]
    scale_x, scale_y = (w / 1000, h / 1000) if coords and max(coords) <= 1000 else (1.0, 1.0)

    def box_of(it):
        b = it.get("bbox_2d") or [0, 0, 0, 0]
        x0, y0, x1, y1 = (int(b[0] * scale_x), int(b[1] * scale_y), int(b[2] * scale_x), int(b[3] * scale_y))
        x0, x1 = sorted((max(0, min(w, x0)), max(0, min(w, x1))))
        y0, y1 = sorted((max(0, min(h, y0)), max(0, min(h, y1))))
        return x0, y0, x1, y1

    boxes, axes, writing = [], [], []
    for it in items:
        kind = str(it.get("type", "")).lower()
        x0, y0, x1, y1 = box_of(it)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        if kind == "box":
            iw, ih = x1 - x0, y1 - y0
            label = frame_bgr[y1 - int(ih * LABEL_H): y1, x0: x0 + int(iw * LABEL_W)]
            corners = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            boxes.append(VBox((x0, y0, x1, y1), corners, label.copy(), bool(it.get("has_drawing")),
                              it.get("label") or ""))
        elif kind == "axes":
            axes.append(VAxes((x0, y0, x1, y1)))
        elif kind in ("expression", "text"):
            m = max(4, int((y1 - y0) * 0.15))                  # small margin, like snippet.py
            bx = (max(0, x0 - m), max(0, y0 - m), min(w, x1 + m), min(h, y1 + m))
            crop = frame_bgr[bx[1]:bx[3], bx[0]:bx[2]].copy()
            text = it.get("latex") if kind == "expression" else it.get("text")
            text = text or ""
            blank = bool(it.get("blank"))
            if kind == "expression":
                stripped = re.sub(r"(\\_|_){2,}\s*$|\\underline\{\s*\}\s*$", "", text).strip()   # "2+2=___"
                blank = blank or stripped != text.strip() or stripped.endswith("=")
                text = stripped
            if kind == "expression" and blank:
                role, hint = "autocomplete", "math"
            else:
                role, hint = "writing", ("math" if kind == "expression" else "text")
            writing.append(VSnippet(bx, crop, hint, role, text or ""))
    for s in writing:                                          # expression just left of axes -> graph
        if s.role == "writing" and s.hint == "math" and any(_left_of_axes(s.bbox, a) for a in axes):
            s.role = "graph"
    return boxes, axes, writing


def _parse_json(raw):
    """The model's JSON array (tolerating code fences and stray text around it)."""
    s = re.sub(r"^```[a-z]*\s*|\s*```$", "", raw.strip())
    start, end = s.find("["), s.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(s[start:end + 1])
    except json.JSONDecodeError:
        try:   # LaTeX backslashes are often not escaped for JSON: escape lone ones and retry
            data = json.loads(re.sub(r'(?<!\\)\\(?![\\"/bfnrtu])', r"\\\\", s[start:end + 1]))
        except json.JSONDecodeError:
            return []
    return [d for d in data if isinstance(d, dict)]
