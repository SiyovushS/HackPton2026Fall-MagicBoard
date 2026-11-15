"""Score the OCR engine against the labelled test set.

Run from the project root:
    .venv/bin/python tests/ocr/eval_ocr.py                 # hinted + auto mode
    .venv/bin/python tests/ocr/eval_ocr.py --mode hint     # only with hints
    .venv/bin/python tests/ocr/eval_ocr.py --save baseline # also save results/baseline.json
    .venv/bin/python tests/ocr/eval_ocr.py --compare baseline   # show what changed vs a saved run

Two scores per item, after normalising both answer and expected text:
    exact      the result matches the expected answer exactly
    char acc   1 - (edit distance / length of expected): partial credit, 1.0 = perfect

Add test items by putting a crop in images/ and an entry in labels.json.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # project root, for ocr_engine
from ocr_engine import OCREngine, match_label  # noqa: E402

# Pretend these artifacts are saved; label crops (ids starting "label_") are matched against them.
KNOWN_ARTIFACTS = ["apple", "house", "cat.jpg", "lec.2", "lecture 1", "maple", "pineapple", "graph 2"]


# ---------- normalisation: ignore differences that don't change meaning ----------
def norm_math(s):
    s = s or ""
    s = re.sub(r"\\(left|right)\b", "", s)            # \left( -> (
    s = re.sub(r"\\[,;:! ]", "", s)                    # LaTeX spacing commands
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace("{", "").replace("}", "")            # x^{2} == x^2
    return re.sub(r"\s+", "", s)


def norm_text(s):
    s = s or ""
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)         # unwrap every \text{...}
    s = s.lower().replace("\\\\", " ")                 # line breaks between lines -> space
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(" .,")


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score(kind, result, expected):
    norm = norm_math if kind == "math" else norm_text
    r, e = norm(result), norm(expected)
    acc = max(0.0, 1 - edit_distance(r, e) / max(len(e), 1))
    return r == e, acc


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hint", "auto", "both"], default="both")
    ap.add_argument("--save", help="save results under results/NAME.json")
    ap.add_argument("--compare", help="compare against results/NAME.json")
    ap.add_argument("--model", help="Hugging Face model id to use instead of the engine's default")
    args = ap.parse_args()

    labels = json.loads((HERE / "labels.json").read_text())
    modes = ["hint", "auto"] if args.mode == "both" else [args.mode]

    t = time.time()
    engine = OCREngine(model=args.model) if args.model else OCREngine()
    print(f"Engine loaded in {time.time() - t:.1f}s\n")

    results = {}
    for item in labels:
        img = cv2.imread(str(HERE / item["image"]))
        for mode in modes:
            t = time.time()
            out = engine.recognize(img, hint=item["kind"] if mode == "hint" else None)
            secs = time.time() - t
            exact, acc = score(item["kind"], out, item["expected"])
            results[f"{item['id']}|{mode}"] = {"id": item["id"], "mode": mode, "kind": item["kind"],
                                                "result": out, "expected": item["expected"],
                                                "exact": exact, "acc": acc, "secs": secs}

    # per-item table
    for mode in modes:
        print(f"=== {mode} mode " + "=" * 60)
        for item in labels:
            r = results[f"{item['id']}|{mode}"]
            mark = "OK " if r["exact"] else ("~  " if r["acc"] >= 0.7 else "X  ")
            out = (r["result"] or "None").replace("\n", " ")
            print(f"{mark}{r['acc']:4.2f}  {item['id']:<20} {out[:60]:<60}  want: {item['expected']}")
        print()

    # summary
    print("=== summary " + "=" * 62)
    print(f"{'mode':<6}{'kind':<6}{'n':>4}{'exact':>9}{'char acc':>10}{'avg s':>8}")
    for mode in modes:
        for kind in ["text", "math", "all"]:
            rs = [r for r in results.values() if r["mode"] == mode and (kind == "all" or r["kind"] == kind)]
            n = len(rs)
            print(f"{mode:<6}{kind:<6}{n:>4}{sum(r['exact'] for r in rs) / n:>9.0%}"
                  f"{sum(r['acc'] for r in rs) / n:>10.2f}{sum(r['secs'] for r in rs) / n:>8.2f}")

    # artifact labels: does fuzzy matching recover the right saved name?
    label_items = [i for i in labels if i["id"].startswith("label_")]
    if label_items and "hint" in modes:
        print(f"\n=== artifact labels: recognize_label() against {KNOWN_ARTIFACTS} " + "=" * 5)
        ok, total = 0, 0
        for item in label_items:
            img = cv2.imread(str(HERE / item["image"]))
            # 1) the name is saved -> must pick it;  2) the name is NOT saved -> must say None
            for names, want in [(KNOWN_ARTIFACTS, item["expected"]),
                                ([n for n in KNOWN_ARTIFACTS if n != item["expected"]], None)]:
                got = engine.recognize_label(img, names)
                ok += got == want
                total += 1
                case = "saved    " if want else "not saved"
                print(f"{'OK ' if got == want else 'X  '}{item['id']:<20} {case} -> {got!r} (want {want!r})")
            free = results[f"{item['id']}|hint"]["result"]
            print(f"   (old way: free reading {free!r} + fuzzy match -> {match_label(free, KNOWN_ARTIFACTS)!r})")
        print(f"label checks passed: {ok}/{total}")

    results_dir = HERE / "results"
    if args.compare:
        old = json.loads((results_dir / f"{args.compare}.json").read_text())
        print(f"\n=== changes vs '{args.compare}' " + "=" * 40)
        changed = False
        for key, r in results.items():
            o = old.get(key)
            if o and (o["exact"] != r["exact"] or abs(o["acc"] - r["acc"]) > 0.05):
                changed = True
                arrow = "BETTER" if r["acc"] > o["acc"] else "WORSE "
                print(f"{arrow} {key:<28} {o['acc']:.2f} -> {r['acc']:.2f}   {o['result']!r} -> {r['result']!r}")
        if not changed:
            print("no item changed")
    if args.save:
        results_dir.mkdir(exist_ok=True)
        (results_dir / f"{args.save}.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\nsaved results/{args.save}.json")


if __name__ == "__main__":
    main()
