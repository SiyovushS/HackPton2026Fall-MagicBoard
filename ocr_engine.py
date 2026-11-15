"""OCR engine: handwritten whiteboard crops -> text / LaTeX, using Qwen3-VL.

One vision-language model (Qwen3-VL, run locally with PyTorch + Hugging Face transformers)
reads both words and math. It uses the Mac GPU (MPS), an NVIDIA GPU (CUDA), or the CPU.
Before reading, OpenCV clean-up removes things the model misreads: box borders and graph
axes, and the "answer blank" in "2+2 = ____".

Chosen by a bake-off on tests/ocr (13 crops): 92% exact vs 62% for the previous
Pix2Text + TexTeller + TrOCR engine. See tests/ocr/eval_ocr.py.

Usage:
    engine = OCREngine()                        # once at startup: loads + warms up the model
    engine.recognize(crop, hint="math")         # known math (next to axes, ends with "=") -> LaTeX
    engine.recognize(crop, hint="text")         # known words (artifact label) -> plain text
    engine.recognize(crop)                      # unknown: words as \\text{...}, math as LaTeX
    engine.has_answer_blank(crop)               # True for "2+2 = ____" (autocomplete trigger)
    engine.recognize_label(crop, saved_names)   # artifact call: which saved name is this? (or None)
    engine.recognize_frames([crop1, crop2, crop3], hint="text")  # same region, several frames: vote
    engine.recognize_many([crop_a, crop_b], ["math", "text"])    # several regions in one batched call

Math with hint="math" that doesn't parse as maths gets a second opinion from GLM-OCR.

`crop` is one region as an OpenCV BGR image (from the board splitter).
The model downloads from Hugging Face on first run.
"""
import difflib
import re
from collections import Counter

import cv2
import numpy as np
import torch
from PIL import Image
from sympy.parsing.latex import parse_latex
from transformers import AutoModelForImageTextToText, AutoProcessor, Qwen3VLForConditionalGeneration

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
FALLBACK_MATH_MODEL = "zai-org/GLM-OCR"  # second opinion when Qwen's math doesn't parse (~0.4 s, ~2 GB)
PAD = 20          # white margin added around every crop before OCR
MAX_SIDE = None   # crops are sent at full size: shrinking to 1024 px cost accuracy on long text lines.
                  # Set a number (e.g. 1024) to cap the size if speed or memory becomes a problem.


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

PROMPTS = {
    "text": ("This is a crop of handwriting on a whiteboard. Transcribe the handwritten words exactly. "
             "If there are several lines, put each on its own line. Output only the transcription."),
    "math": ("This is a crop of handwritten math on a whiteboard. Transcribe it as LaTeX, without $ "
             "delimiters. Output only the LaTeX."),
    None: ("This is a crop of handwriting on a whiteboard. Transcribe everything written. Write words as "
           "plain text and math as LaTeX wrapped in $...$. Put each line of writing on its own line. "
           "Output only the transcription."),
    "label": ("This crop shows a short handwritten label on a whiteboard. Which of these is written, "
              "letter for letter: {options}? Reply with that option exactly as listed, nothing else."),
}
# Most answers are short; a lower cap stops runaway output early and saves time.
MAX_TOKENS = {"label": 16, "text": 96, "math": 160, None: 200}


GREEK =["alpha", "beta", "gamma", "delta", "theta", "lambda", "mu", "pi", "sigma", "phi", "omega"]


def clean_latex(latex):
    """Tidy model output: drop \\left/\\right, write Greek names as symbols, fix spacing.

    e.g. "\\ln \\left( 3 \\right)+3+pi" -> "\\ln(3)+3+\\pi"
    """
    s = latex or ""
    s = re.sub(r"^\s*\${1,2}|\${1,2}\s*$", "", s.strip())  # $$ ... $$ around the whole answer
    s = re.sub(r"\\underline\s*\{\s*(\{\s*\})?\s*\}", "", s)  # empty answer blank: \underline{{}}
    s = re.sub(r"\\(left|right)\s*\.", "", s)              # invisible delimiters \left. \right.
    s = re.sub(r"\\(left|right)\s*", "", s)                # \left( -> (
    for g in GREEK:                                        # "pi" written as letters -> \pi
        s = re.sub(rf"(?<![A-Za-z\\]){g}(?![A-Za-z])", rf"\\{g}", s)
    s = re.sub(r"\s*([()\[\]+=\-^_,{}])\s*", r"\1", s)    # no spaces around brackets/operators
    s = re.sub(r"(?<=\d)\s+(?=\d)", "", s)                 # "7 0" -> "70"
    return re.sub(r"\s+", " ", s).strip()


def latex_parses(latex):
    """True if SymPy can parse the LaTeX as maths (a trailing "=" for autocomplete is allowed).

    Catches broken output. It cannot catch output that is valid maths but the wrong maths.
    """
    s = re.sub(r"=\s*$", "", latex or "").strip()
    if not s:
        return False
    try:
        for side in s.split("="):
            parse_latex(side, backend="lark")
        return True
    except Exception:
        return False


def normalise_reading(s):
    """Key for comparing two readings of the same region: ignore case, spaces, braces, punctuation."""
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s or "").lower()
    return re.sub(r"[\s{}.,;:!?]", "", s)


def vote(readings):
    """Most common reading (by normalise_reading); ties go to the earliest. None if all empty."""
    readings = [r for r in readings if r]
    if not readings:
        return None
    counts = Counter(normalise_reading(r) for r in readings)
    best = max(counts.values())
    return next(r for r in readings if counts[normalise_reading(r)] == best)


def label_key(s):
    """Compare labels ignoring case, spaces and punctuation: "Cat.JPG" == "cat jpg"."""
    s = re.sub(r"\\text\{(.*)\}", r"\1", s or "")
    return re.sub(r"[^a-z0-9]", "", s.lower())


def close_labels(text, known_names, cutoff=0.6):
    """Known names that look like `text`, closest first. Numbers must match exactly."""
    target = label_key(text)
    if not target:
        return []
    digits = re.sub(r"\D", "", target)
    scored = [(difflib.SequenceMatcher(None, target, label_key(n)).ratio(), n) for n in known_names
              if re.sub(r"\D", "", label_key(n)) == digits]  # "lecture 2" must never become "lecture 1"
    return [n for score, n in sorted(scored, reverse=True) if score >= cutoff]


def match_label(text, known_names, cutoff=0.6):
    """Fuzzy-match an OCR'd label to the closest known name, e.g. "cait-jpg" -> "cat.jpg".

    Text only, no image: it cannot tell "hoose" (a misread "house") from "apple" (a real,
    unsaved name close to "maple"). OCREngine.recognize_label() resolves that with the image.
    """
    close = close_labels(text, known_names, cutoff)
    return close[0] if close else None


BOARD = "auto"  # "auto", "white" (dark marker on a light board) or "black" (light chalk on a dark board)


def ink_mask(crop_bgr, relative=0.08, floor=12, board=None):
    """Pixels that stand out from the board around them: the writing.

    Whiteboards have ink DARKER than the board, chalkboards have chalk LIGHTER than it.
    board="auto" (default) measures both and keeps whichever direction has more stroke-like
    ink, so the same code works on both (a dark-only version found zero ink on chalk).

    "Stands out" is relative: at least `relative` (8%) different from the local board
    brightness and at least `floor` grey levels. A fixed threshold missed faint webcam ink
    (~10% darker). Noisy, white and unevenly lit empty boards give zero ink pixels.
    """
    board = board or BOARD
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    background = cv2.medianBlur(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY), 21).astype(np.float32)
    needed = np.maximum(floor, relative * np.maximum(background, 255 - background))
    dark = ((background - gray) > needed).astype(np.uint8) * 255   # marker on whiteboard
    light = ((gray - background) > needed).astype(np.uint8) * 255  # chalk on chalkboard
    if board == "white":
        return dark
    if board == "black":
        return light
    return dark if stroke_pixels(dark) >= stroke_pixels(light) else light


def stroke_pixels(mask, min_speck=15):
    """Ink pixels in blobs of at least `min_speck` pixels (dust and noise ignored)."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    return int(areas[areas >= min_speck].sum())


def remove_edge_lines(crop_bgr, band_frac=0.18):
    """Erase box borders, single box sides and graph-axes corners from a crop.

    Each connected stroke that is large relative to the crop is measured:
      perim     share of its ink lying near its own bounding box's edge (outline-like)
      straight  thickness / length of the stroke (small = a straight line)
    and erased when it matches one of these (thresholds tuned on tests/ocr):
      box:        spans >= 70% of both width and height, perim >= 0.7
                  (only its outer band is erased, so writing touching the box survives)
      side/axes:  spans >= 80% of height (or width), sits at the crop's edge,
                  and is straight (<= 0.1) or outline-like (perim >= 0.7)
    Kept on purpose: "x^2" in a tight crop (perim 0.49), the "l" in "ln" (spans only ~70%),
    an integral sign (straight 0.17, perim 0.61), a fraction bar (not at an edge).
    """
    h, w = crop_bgr.shape[:2]
    ink = ink_mask(crop_bgr)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))  # join broken strokes
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    erase = np.zeros((h, w), bool)
    ys, xs = np.mgrid[0:h, 0:w]
    for i in range(1, n):
        bx, by, bw, bh, _ = stats[i]
        span_w, span_h = bw / w, bh / h
        if max(span_w, span_h) < 0.7:
            continue
        comp = labels == i
        outer = ((xs < bx + bw * band_frac) | (xs > bx + bw - bw * band_frac) |
                 (ys < by + bh * band_frac) | (ys > by + bh - bh * band_frac))
        perim = (comp & outer).sum() / comp.sum()
        ev = np.sort(np.linalg.eigvalsh(np.cov(np.column_stack(np.nonzero(comp)).T.astype(float))))
        straight = np.sqrt(max(ev[0], 0)) / max(np.sqrt(ev[1]), 1e-6)
        at_side = bx + bw <= 0.3 * w or bx >= 0.7 * w
        at_top_bottom = by + bh <= 0.3 * h or by >= 0.7 * h
        line_like = straight <= 0.1 or perim >= 0.7
        if span_w >= 0.7 and span_h >= 0.7 and perim >= 0.7:  # whole box
            erase |= comp & outer
        elif (span_h >= 0.8 and at_side and line_like) or (span_w >= 0.8 and at_top_bottom and line_like):
            erase |= comp  # single side or axes
    if not erase.any():
        return crop_bgr
    erase = cv2.dilate(erase.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0  # cover stroke edges
    cleaned = crop_bgr.copy()
    cleaned[erase] = cv2.medianBlur(crop_bgr, 21)[erase]  # paint with the board's own colour
    return cleaned
TROCR_MODEL = "microsoft/trocr-base-handwritten"  # "-large-" was ~2x slower and no more accurate


def find_answer_blank(crop_bgr):
    """Find an answer blank ("2+2 = ____"): a long flat line at the END of the writing, low down.

    Returns the blank's connected-component mask, or None. Short strokes like "=" or "-"
    don't qualify, nor do centred lines like a fraction bar, nor a minus sign of the line
    below caught in the margin (it isn't at the end of the writing).
    """
    h, w = crop_bgr.shape[:2]
    ink = cv2.morphologyEx(ink_mask(crop_bgr), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    real = [i for i in range(1, n) if stats[i][cv2.CC_STAT_AREA] >= 15]   # ignore dust specks
    if not real:
        return None
    ink_right = max(stats[i][0] + stats[i][2] for i in real)      # where the writing ends
    best = None
    for i in range(1, n):
        bx, by, bw, bh, _ = stats[i]
        flat = bw >= 4 * bh
        at_end = bx + bw >= ink_right - 0.05 * w                  # the blank finishes the line
        if flat and at_end and bw >= 0.15 * w and bx >= 0.45 * w and by + bh / 2 >= 0.45 * h:
            if best is None or bw > stats[best][2]:
                best = i
    return None if best is None else labels == best


class OCREngine:
    def __init__(self, model: str = MODEL, device: str = None, math_fallback: bool = True):
        self.device = device or pick_device()
        self.dtype = torch.float32 if self.device == "cpu" else torch.float16
        self.processor = AutoProcessor.from_pretrained(model)
        self.processor.tokenizer.padding_side = "left"  # needed for batched generation
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model, dtype=self.dtype).to(self.device).eval()
        self.math_fallback = math_fallback
        self._glm = None  # GLM-OCR, loaded on first use (only if Qwen's math fails to parse)
        self._warm_up()

    def _warm_up(self):
        # The first call is slow (graph compilation); do it before going live.
        sample = np.full((60, 200, 3), 255, np.uint8)
        cv2.putText(sample, "x+1", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        self._ask_many([sample], [PROMPTS["math"]], MAX_TOKENS["math"])

    # ---------- main entry points ----------
    def recognize(self, crop_bgr, hint=None, debug=False):
        """Return the reading as one string (lines joined with " \\\\ "), or None.

        hint="math" -> LaTeX;  hint="text" -> plain text;  None -> LaTeX with words in \\text{...}.
        """
        return join_lines(self.recognize_lines(crop_bgr, hint=hint, debug=debug))

    def recognize_lines(self, crop_bgr, hint=None, debug=False):
        """Like recognize(), but returns a list with one string per line of writing."""
        return self.recognize_many([crop_bgr], [hint], debug=debug)[0]

    def recognize_many(self, crops, hints=None, batch_size=1, debug=False):
        """Read several crops, `batch_size` at a time in one model call. Returns a list of line-lists.

        On the Mac GPU, batching was measured SLOWER (8 math crops: 13.2 s one at a time vs
        16.0 s four at a time, same results), so the default is 1. Try 4 on an NVIDIA GPU.
        """
        hints = hints or [None] * len(crops)
        prepared = [self._prepare(c) for c in crops]
        results = [[] for _ in crops]
        todo = [i for i, p in enumerate(prepared) if p is not None]  # skip empty crops
        for k in range(0, len(todo), batch_size):
            idx = todo[k:k + batch_size]
            raws = self._ask_many([prepared[i] for i in idx], [PROMPTS[hints[i]] for i in idx],
                                  max(MAX_TOKENS[hints[i]] for i in idx))
            for i, raw in zip(idx, raws):
                if debug:
                    print(f"    raw={raw!r}")
                results[i] = self._parse(raw, hints[i], prepared[i], debug)
        return results

    def recognize_frames(self, crops, hint=None, max_reads=3, debug=False):
        """Read the same region from several video frames and keep the answer that repeats.

        Stops as soon as two readings agree, so it usually costs 2 reads, at most `max_reads`.
        Helps with random misreads that differ between frames. It does NOT fix systematic ones:
        on near-identical simulated frames, "Basic" came back as Basic / Balsie / Bolsie.
        """
        readings = []
        for crop in crops[:max_reads]:
            readings.append(self.recognize(crop, hint=hint))
            if debug:
                print(f"    read {len(readings)}: {readings[-1]!r}")
            keys = Counter(normalise_reading(r) for r in readings if r)
            if keys and max(keys.values()) >= 2:
                break
        return vote(readings)

    def recognize_label(self, crop_bgr, known_names, debug=False):
        """Which saved artifact name is written here? Returns one of `known_names`, or None.

        1. Read the label freely. If that is a saved name (ignoring case/punctuation), done.
        2. Otherwise find saved names that look similar ("hoose" ~ "house", "apple" ~ "maple")
           and ask the model to choose between its own reading and those lookalikes, with the
           image. Picking its own reading means "not saved" -> None.
        (Asking it to pick straight from the whole list was tested and fails: it picks a
        lookalike, e.g. "maple" for "apple", even when the right name is not saved.)
        """
        prepared = self._prepare(crop_bgr) if known_names else None
        if prepared is None:
            return None
        free = " ".join(self._parse(self._ask_many([prepared], [PROMPTS["text"]], MAX_TOKENS["text"])[0], "text", prepared))
        exact = [n for n in known_names if label_key(n) == label_key(free)]
        if debug:
            print(f"    free reading={free!r} exact={exact}")
        if exact:
            return exact[0]
        lookalikes = close_labels(free, known_names)[:4]
        if not lookalikes:
            return None
        options = [free] + lookalikes
        answer = self._ask_many([prepared], [PROMPTS["label"].format(options=", ".join(f'"{o}"' for o in options))],
                                MAX_TOKENS["label"])[0]
        answer = answer.strip().splitlines()[0].strip("\"'.` ") if answer.strip() else ""
        if debug:
            print(f"    options={options} answer={answer!r}")
        chosen = [n for n in lookalikes if label_key(n) == label_key(answer)]
        return chosen[0] if chosen else None

    @staticmethod
    def has_answer_blank(crop_bgr):
        """True if the crop ends in an answer blank, e.g. "2+2 = ____" (autocomplete trigger)."""
        return find_answer_blank(remove_edge_lines(crop_bgr)) is not None

    @staticmethod
    def clean_crop(crop_bgr):
        """Erase box borders, axes and the answer blank, which models misread as symbols."""
        crop_bgr = remove_edge_lines(crop_bgr)
        blank = find_answer_blank(crop_bgr)
        if blank is not None:
            erase = cv2.dilate(blank.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            crop_bgr = crop_bgr.copy()
            crop_bgr[erase] = cv2.medianBlur(crop_bgr, 21)[erase]
        return crop_bgr

    # ---------- internals ----------
    @classmethod
    def _prepare(cls, crop_bgr):
        """Clean-up before reading; None if there is nothing to read (blank, or only a frame)."""
        if not has_ink(crop_bgr):  # empty/erased region: models hallucinate on blank images
            return None
        crop_bgr = cls.clean_crop(crop_bgr)
        return crop_bgr if has_ink(crop_bgr) else None

    def _parse(self, raw, hint, prepared, debug=False):
        lines = [line for line in (strip_wrappers(x) for x in raw.splitlines()) if line]
        if hint == "text":
            return lines
        if hint == "math":
            # one expression per written line: joining "ax^2+bx+c" and "(x-q0)(x-q1)" would
            # wrongly read as a single product
            latexes = [clean_latex(line) for line in lines]
            if self.math_fallback and len(latexes) == 1 and (
                    latex_looks_broken(latexes[0]) or not latex_parses(latexes[0])):
                second = clean_latex(self._ask_glm(prepared))  # second opinion from GLM-OCR
                if debug:
                    print(f"    qwen={latexes[0]!r} doesn't parse; glm-ocr={second!r}")
                if not latex_looks_broken(second) and latex_parses(second):
                    latexes = [second]
            return [x for x in latexes if not latex_looks_broken(x)]
        return [mixed_to_latex(line) for line in lines]

    @staticmethod
    def _to_image(crop_bgr):
        padded = cv2.copyMakeBorder(crop_bgr, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        if MAX_SIDE and max(padded.shape[:2]) > MAX_SIDE:
            scale = MAX_SIDE / max(padded.shape[:2])
            padded = cv2.resize(padded, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return Image.fromarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))

    def _ask_many(self, crops, instructions, max_tokens):
        conversations = [[{"role": "user", "content": [{"type": "image", "image": self._to_image(c)},
                                                       {"type": "text", "text": t}]}]
                         for c, t in zip(crops, instructions)]
        inputs = self.processor.apply_chat_template(conversations, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt",
                                                    processor_kwargs={"padding": True}).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        return [t.strip() for t in self.processor.batch_decode(new_tokens, skip_special_tokens=True)]

    def _ask_glm(self, crop_bgr):
        if self._glm is None:
            processor = AutoProcessor.from_pretrained(FALLBACK_MATH_MODEL)
            model = AutoModelForImageTextToText.from_pretrained(FALLBACK_MATH_MODEL, dtype=self.dtype)
            self._glm = (processor, model.to(self.device).eval())
        processor, model = self._glm
        messages = [{"role": "user", "content": [{"type": "image", "image": self._to_image(crop_bgr)},
                                                 {"type": "text", "text": "Formula Recognition:"}]}]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt").to(self.device)
        inputs.pop("token_type_ids", None)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_TOKENS["math"], do_sample=False)
        return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def join_lines(lines):
    return " \\\\ ".join(lines) if lines else None


# ---------- output helpers ----------
def strip_wrappers(s):
    """Remove code fences and $ / \\( \\) / \\[ \\] wrapped around a whole line."""
    s = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", s.strip())
    s = re.sub(r"^\\\(|\\\)$|^\\\[|\\\]$", "", s.strip())
    return s.strip()


def mixed_to_latex(line):
    """"Find $x$ where $x^2=4$" -> "\\text{Find } x \\text{ where } x^2=4"."""
    parts = re.split(r"(\$\$.*?\$\$|\$.*?\$)", line)
    out = []
    for p in parts:
        if p.startswith("$"):
            out.append(clean_latex(p.strip("$")))
        elif p.strip():
            out.append(r"\text{" + p + "}")
    return " ".join(out)


def has_ink(crop_bgr, min_pixels=60, min_speck=15):
    """True if the crop has real marks: at least `min_pixels` of ink in strokes (specks ignored).

    Counts pixels, not a share of the crop, so a small label in a big crop still counts
    (a share-based check missed a webcam label: 0.13% of a whole-box crop).
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return False
    return stroke_pixels(ink_mask(crop_bgr), min_speck) >= min_pixels


def latex_looks_broken(latex):
    """Catch obvious garbage: runaway repetition, empty structures, unbalanced braces."""
    if not latex or not latex.strip():
        return True
    if "\\newcommand" in latex:                         # hallucinated preamble
        return True
    if re.search(r"(\\cdot\s*){5,}", latex):            # \\cdot\\cdot\\cdot...
        return True
    if re.search(r"(\b\d\s+){6,}", latex):             # 1 0 0 0 0 0 ...
        return True
    if re.search(r"\\frac\{\s*\}\s*\{\s*\}", latex):   # \\frac{}{}
        return True
    return latex.count("{") != latex.count("}")       # unbalanced braces


if __name__ == "__main__":
    # Quick test:  python ocr_engine.py path/to/crop.png [math|text]
    import sys

    if len(sys.argv) < 2:
        sys.exit("usage: python ocr_engine.py IMAGE [math|text]")
    image = cv2.imread(sys.argv[1])
    if image is None:
        sys.exit(f"could not read {sys.argv[1]}")
    engine = OCREngine()
    print(engine.recognize(image, hint=sys.argv[2] if len(sys.argv) > 2 else None, debug=True))
