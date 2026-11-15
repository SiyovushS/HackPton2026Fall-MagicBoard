"""What the app DOES with what it reads on the board.

    answer = solve_autocomplete("2+2=")              # "4"
    curve = graph_points("x^{2}+2x+3", axes_bbox)    # pixel points to draw inside the drawn axes
    store = ArtifactStore("artifacts")
    store.save("apple", crop_bgr)                    # DEFINE box: keep its drawing
    image = store.load("apple")                      # CALL box: the saved drawing (BGRA), or None
"""
import re
from pathlib import Path

import cv2
import numpy as np
import sympy
from sympy.parsing.latex import parse_latex

from ocr_engine import clean_latex, ink_mask, label_key, match_label


# ---------------- autocomplete ----------------
def solve_autocomplete(latex):
    """The answer for an expression ending in '=' (e.g. "2+2=" -> "4"), or None if it can't be
    computed. Exact when possible (fractions, sqrt), with a decimal for irrational results."""
    expr = _to_sympy(re.sub(r"=\s*$", "", clean_latex(latex or "")))
    if expr is None or expr.free_symbols:
        return None                                    # unknowns (x, y): nothing to compute
    try:
        exact = sympy.nsimplify(sympy.simplify(expr))
        if exact.is_number and not exact.is_Rational and not exact.has(sympy.pi, sympy.E):
            value = complex(sympy.N(exact))
            if abs(value.imag) > 1e-9:
                return None
            return _round(value.real)
        if exact.is_Rational:
            return str(exact)
        return f"{sympy.latex(exact)} \\approx {_round(float(sympy.N(exact)))}"
    except Exception:
        return None


def _round(v):
    return f"{v:.4g}"


# ---------------- graphing ----------------
def graph_points(latex, axes_bbox, corner=None, samples=120):
    """Points (x, y in frame pixels) of y = f(x) drawn inside the professor's axes.

    The axes' corner is the origin; x runs along the horizontal arm from 0 to 5, and the y scale
    is chosen so the curve fits the vertical arm. Returns None if the expression can't be plotted.
    """
    expr = _to_sympy(re.sub(r"^\s*y\s*=\s*", "", clean_latex(latex or "")))
    if expr is None:
        return None
    symbols = sorted(expr.free_symbols, key=lambda s: s.name)
    if len(symbols) != 1:
        return None
    f = sympy.lambdify(symbols[0], expr, "numpy")
    x0, y0, x1, y1 = axes_bbox
    ox, oy = corner if corner else (x0, y1)                   # origin: bottom-left by default
    xs = np.linspace(0, 5, samples)
    with np.errstate(all="ignore"):
        ys = np.asarray(f(xs), dtype=float) * np.ones_like(xs)
    ok = np.isfinite(ys)
    if ok.sum() < 2:
        return None
    top = max(np.nanmax(np.abs(ys[ok])), 1e-6)
    px = ox + xs / 5 * (x1 - ox) * 0.95
    py = oy - ys / top * (oy - y0) * 0.9
    return np.stack([px[ok], py[ok]], axis=1).astype(np.int32)


def _to_sympy(latex):
    if not latex:
        return None
    s = latex.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\log_\{?(\d+)\}?", r"\\log_{\1}", s)
    try:
        expr = parse_latex(s, backend="lark")
    except Exception:
        return None
    if hasattr(expr, "children"):
        # Ambiguous: "\ln(3)+\log_{10}(40)" also parses as ln(log10(40) + 3). The intended reading
        # is the least nested one, so take the shallowest expression tree.
        options = [c for c in expr.children if isinstance(c, sympy.Basic)]
        expr = min(options, key=_depth) if options else None
    return expr if isinstance(expr, sympy.Basic) else None


def _depth(e):
    return 1 + max((_depth(a) for a in e.args), default=0)


# ---------------- artifacts ----------------
class ArtifactStore:
    """Saved drawings, one PNG per label (with a transparent background)."""

    def __init__(self, folder="artifacts"):
        self.folder = Path(folder)
        self.folder.mkdir(exist_ok=True)
        self._cache = {}

    def names(self):
        return [p.stem for p in self.folder.glob("*.png")]

    def save(self, label, crop_bgr, keep=None):
        """Save a DEFINE box's inside: ink kept, board made transparent, trimmed to the drawing.

        keep: optional mask of the ink to keep (e.g. only ink that is new since the session
        started, so an old mark that happened to be inside the box isn't copied along)."""
        key = label_key(label)
        if not key or crop_bgr is None or crop_bgr.size == 0:
            return None
        ink = ink_mask(crop_bgr)
        if keep is not None:
            ink[cv2.dilate(keep, np.ones((5, 5), np.uint8)) == 0] = 0
        ink = cv2.dilate(ink, np.ones((3, 3), np.uint8))
        ys, xs = np.nonzero(ink)
        if len(xs) < 20:
            return None
        pad = 6
        y0, y1 = max(0, ys.min() - pad), min(crop_bgr.shape[0], ys.max() + pad)
        x0, x1 = max(0, xs.min() - pad), min(crop_bgr.shape[1], xs.max() + pad)
        rgba = cv2.cvtColor(crop_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2BGRA)
        rgba[..., 3] = cv2.GaussianBlur(ink[y0:y1, x0:x1], (3, 3), 0)
        path = self.folder / f"{key}.png"
        cv2.imwrite(str(path), rgba)
        self._cache.pop(key, None)
        return path

    def load(self, label):
        """The saved drawing for this label (fuzzy match, numbers exact), or None."""
        name = match_label(label or "", self.names(), cutoff=0.75)
        if name is None:
            return None
        if name not in self._cache:
            self._cache[name] = cv2.imread(str(self.folder / f"{name}.png"), cv2.IMREAD_UNCHANGED)
        return self._cache[name]
