"""Score the whole pipeline (snipping + reading) on saved board images.

Run from the project root:
    .venv/bin/python tests/board/eval_board.py                       # all boards
    .venv/bin/python tests/board/eval_board.py --save NAME           # also save results/NAME.json
    .venv/bin/python tests/board/eval_board.py --compare NAME        # summary vs a saved run
    .venv/bin/python tests/board/eval_board.py --only chalk_a        # boards whose id contains this
    .venv/bin/python tests/board/eval_board.py --pictures            # also write pictures/<id>.jpg
    .venv/bin/python tests/board/eval_board.py --method vlm          # how regions are found (see below)

Methods:
  snippets   snippet.py's rules find the regions, each crop is read (the current pipeline)
  vlm        Qwen3-VL finds AND reads everything in one call on the whole board (vlm_layout.py)
  vlm+read   Qwen3-VL finds the regions, then each crop is read separately

For each board in labels.json:
  boxes    expected boxes found, of the right kind (DEFINE = has content, CALL = empty), label read
  writing  expected expressions found (>= 75% character accuracy), exact, and with the right role
           (autocomplete / graph / writing). "optional" items don't count when missed.
  junk     things read that match nothing expected (smudges, edges, hallucinations)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))   # project root
sys.path.insert(0, str(HERE.parent / "ocr"))  # for the shared scoring helpers
from eval_ocr import edit_distance, norm_math, norm_text  # noqa: E402
from ocr_engine import OCREngine, match_label  # noqa: E402
from people import PersonFinder  # noqa: E402
from snippet import draw_snippets, snip_board  # noqa: E402

FOUND = 0.75   # character accuracy needed to count an expected expression as found


def accuracy(kind, got, want):
    norm = norm_math if kind == "math" else norm_text
    g, w = norm(got), norm(want)
    return max(0.0, 1 - edit_distance(g, w) / max(len(w), 1))


def score_board(item, boxes, box_labels, writing, readings):
    out = {"boxes": [], "writing": [], "junk": []}
    # --- boxes: match expected to found by kind, then check the label
    free = list(range(len(boxes)))
    for exp in item["boxes"]:
        hit = next((i for i in free if ("DEFINE" if boxes[i].has_content else "CALL") == exp["kind"]), None)
        if hit is None:
            hit = free[0] if free else None          # found, but the wrong kind
        if hit is None:
            out["boxes"].append({"expected": exp, "found": False})
            continue
        free.remove(hit)
        kind = "DEFINE" if boxes[hit].has_content else "CALL"
        label_ok = exp["label"] is None or match_label(box_labels[hit] or "", [exp["label"]], cutoff=0.8) is not None
        out["boxes"].append({"expected": exp, "found": True, "kind_ok": kind == exp["kind"],
                             "label": box_labels[hit], "label_ok": label_ok})
    out["extra_boxes"] = len(free)
    # --- writing: best one-to-one matches by accuracy
    pairs = sorted(((accuracy(e["kind"], readings[j] or "", e["text"]), i, j)
                    for i, e in enumerate(item["writing"]) for j in range(len(writing))), reverse=True)
    used_e, used_s = set(), set()
    for acc, i, j in pairs:
        if acc < FOUND or i in used_e or j in used_s:
            continue
        used_e.add(i)
        used_s.add(j)
        e = item["writing"][i]
        out["writing"].append({"expected": e["text"], "optional": e.get("optional", False), "found": True,
                               "exact": acc == 1.0, "acc": round(acc, 2), "read": readings[j],
                               "role_ok": writing[j].role == e["role"], "role": writing[j].role})
    for i, e in enumerate(item["writing"]):
        if i not in used_e:
            out["writing"].append({"expected": e["text"], "optional": e.get("optional", False), "found": False})
    for j, s in enumerate(writing):
        r = readings[j]
        if j not in used_s and r and r != "(nothing read)":
            out["junk"].append({"read": r, "role": s.role, "bbox": s.bbox})
    return out


def summarise(results):
    def count(key, cond):
        return sum(1 for r in results.values() for x in r[key] if cond(x))
    req = lambda x: not x.get("optional")
    return {
        "boxes_expected": count("boxes", lambda x: True),
        "boxes_found": count("boxes", lambda x: x["found"]),
        "boxes_kind_ok": count("boxes", lambda x: x["found"] and x["kind_ok"]),
        "boxes_label_ok": count("boxes", lambda x: x["found"] and x["label_ok"]),
        "extra_boxes": sum(r["extra_boxes"] for r in results.values()),
        "writing_expected": count("writing", req),
        "writing_found": count("writing", lambda x: req(x) and x["found"]),
        "writing_exact": count("writing", lambda x: req(x) and x["found"] and x["exact"]),
        "writing_role_ok": count("writing", lambda x: req(x) and x["found"] and x["role_ok"]),
        "junk": sum(len(r["junk"]) for r in results.values()),
        "seconds_per_board": round(sum(r["seconds"] for r in results.values()) / max(1, len(results)), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save")
    ap.add_argument("--compare")
    ap.add_argument("--only")
    ap.add_argument("--pictures", action="store_true")
    ap.add_argument("--method", choices=["snippets", "vlm", "vlm+read"], default="snippets")
    args = ap.parse_args()

    labels = [b for b in json.loads((HERE / "labels.json").read_text()) if not args.only or args.only in b["id"]]
    t = time.time()
    engine, people = OCREngine(), PersonFinder()
    print(f"Models loaded in {time.time() - t:.1f}s\n")

    results = {}
    for item in labels:
        img = cv2.imread(str(HERE / item["image"]))
        t = time.time()
        if args.method == "snippets":
            boxes, axes, writing = snip_board(img, ignore=people.mask(img))
        else:
            from vlm_layout import find_with_vlm
            boxes, axes, writing = find_with_vlm(engine, img)
        if args.method == "vlm":                       # the model's own readings, no second pass
            box_labels = [b.text for b in boxes]
            readings = [s.text or "(nothing read)" for s in writing]
        else:
            box_labels = [engine.recognize(b.label_crop, hint="text") for b in boxes]
            readings = [engine.recognize(s.crop, hint=s.hint) or "(nothing read)" for s in writing]
        r = score_board(item, boxes, box_labels, writing, readings)
        r["seconds"] = round(time.time() - t, 1)
        results[item["id"]] = r
        if args.pictures:
            (HERE / "pictures").mkdir(exist_ok=True)
            cv2.imwrite(str(HERE / "pictures" / f"{item['id']}.jpg"), draw_snippets(img, boxes, axes, writing, readings))

        print(f"=== {item['id']}  ({r['seconds']}s)")
        for b in r["boxes"]:
            e = b["expected"]
            if not b["found"]:
                print(f"  X  box {e['kind']} '{e['label']}': not found")
            else:
                mark = "OK " if b["kind_ok"] and b["label_ok"] else "~  "
                print(f"  {mark}box {e['kind']} '{e['label']}': found as "
                      f"{'right' if b['kind_ok'] else 'WRONG'} kind, label read {b['label']!r}")
        if r["extra_boxes"]:
            print(f"  X  {r['extra_boxes']} unexpected box(es)")
        for w in r["writing"]:
            opt = " (optional)" if w["optional"] else ""
            if not w["found"]:
                print(f"  {'-  ' if w['optional'] else 'X  '}{w['expected']}{opt}: not found")
            else:
                mark = "OK " if w["exact"] and w["role_ok"] else "~  "
                role = "" if w["role_ok"] else f"  [role {w['role']}, wanted other]"
                print(f"  {mark}{w['expected']}{opt}: read {w['read']!r} ({w['acc']:.2f}){role}")
        for j in r["junk"]:
            print(f"  !  junk [{j['role']}]: {j['read'][:60]!r}")
        print()

    s = summarise(results)
    print("=== summary " + "=" * 50)
    print(f"boxes:   {s['boxes_found']}/{s['boxes_expected']} found, {s['boxes_kind_ok']} right kind, "
          f"{s['boxes_label_ok']} label right, {s['extra_boxes']} unexpected")
    print(f"writing: {s['writing_found']}/{s['writing_expected']} found, {s['writing_exact']} exact, "
          f"{s['writing_role_ok']} right role")
    print(f"junk:    {s['junk']} reads matching nothing")
    print(f"time:    {s['seconds_per_board']}s per board")

    results_dir = HERE / "results"
    if args.compare:
        old = json.loads((results_dir / f"{args.compare}.json").read_text())["summary"]
        print(f"\n=== vs '{args.compare}': " + ", ".join(f"{k} {old[k]} -> {s[k]}" for k in s if old.get(k) != s[k]) or "no change")
    if args.save:
        results_dir.mkdir(exist_ok=True)
        (results_dir / f"{args.save}.json").write_text(json.dumps({"summary": s, "boards": results}, indent=2,
                                                                   default=str, ensure_ascii=False))
        print(f"\nsaved results/{args.save}.json")
    people.close()


if __name__ == "__main__":
    main()
