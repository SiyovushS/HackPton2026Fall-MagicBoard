"""Run the combined pipeline (board_reader.py) on a recorded lecture and score the end result.

    .venv/bin/python tests/board/eval_video.py tests/board/videos/board_test.json

Plays the video at 3 updates per second. Whenever the board has changed and settled, the
board is analysed (Qwen3-VL + new-since-start check + crop reading). The video is paused during
an analysis, so the log shows the analysis time separately: in a live session the result would
appear that many seconds after the board settled.
"""
import json
import sys
import time
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "ocr"))
from board_reader import BoardReader  # noqa: E402
from eval_ocr import edit_distance, norm_math, norm_text  # noqa: E402
from ocr_engine import OCREngine, match_label  # noqa: E402
from people import PersonFinder  # noqa: E402


def accuracy(kind, got, want):
    norm = norm_math if kind == "math" else norm_text
    g, w = norm(got or ""), norm(want)
    return max(0.0, 1 - edit_distance(g, w) / max(len(w), 1))


def main():
    key = json.loads(Path(sys.argv[1]).read_text())
    cap = cv2.VideoCapture(str(ROOT / key["video"]))
    fps = cap.get(cv2.CAP_PROP_FPS)
    engine, people = OCREngine(), PersonFinder()
    reader = BoardReader(engine, people)
    step, i, wall = max(1, int(round(fps / 3))), 0, time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                now = i / fps
                reader.update(frame, now)
                if reader.should_analyse(now):
                    a = reader.analyse(now)
                    print(f"t={now:6.1f}s  analysis took {a.seconds:4.1f}s: kept {len(a.kept)}, dropped {len(a.dropped)}")
                    for it in a.kept:
                        what = f"label {it.label!r}, {'DEFINE' if it.has_content else 'CALL'}" if it.kind == "box" else it.text
                        print(f"     keep {it.kind:10s} {it.role:12s} {what!r:40s} ({it.new_share:.0%} new)")
                    for it, why in a.dropped:
                        print(f"     drop {it.kind:10s} {(it.label or it.text)!r:40s} {why}")
            i += 1
    finally:
        people.close()

    items = reader.items
    print(f"\n=== end of video ({i / fps:.0f}s of video, {time.time() - wall:.0f}s to process, "
          f"{len(reader.history)} analyses, avg {sum(a.seconds for a in reader.history) / max(1, len(reader.history)):.1f}s each)")
    boxes = [it for it in items if it.kind == "box"]
    used = set()
    box_ok = 0
    for exp in key["boxes"]:
        hit = next((j for j, b in enumerate(boxes) if j not in used and
                    ("DEFINE" if b.has_content else "CALL") == exp["kind"] and
                    match_label(b.label or "", [exp["label"]], cutoff=0.8)), None)
        if hit is None:
            hit = next((j for j, b in enumerate(boxes) if j not in used), None)
            print(f"  {'X ' if hit is None else '~ '} box {exp['kind']} {exp['label']!r}: "
                  + ("not found" if hit is None else f"found as {'DEFINE' if boxes[hit].has_content else 'CALL'} {boxes[hit].label!r}"))
        else:
            box_ok += 1
            print(f"  OK box {exp['kind']} {exp['label']!r}")
        if hit is not None:
            used.add(hit)
    n_axes = sum(it.kind == "axes" for it in items)
    print(f"  {'OK' if n_axes == key['axes'] else 'X '} axes: {n_axes} found, {key['axes']} expected")
    writing = [it for it in items if it.kind in ("expression", "text")]
    taken, found, exact, role_ok = set(), 0, 0, 0
    for exp in key["writing"]:
        best = max(((accuracy(exp["kind"], w.text, exp["text"]), j) for j, w in enumerate(writing) if j not in taken),
                   default=(0, None))
        if best[1] is not None and best[0] >= 0.75:
            w = writing[best[1]]
            taken.add(best[1])
            found += 1
            exact += best[0] == 1.0
            role_ok += w.role == exp["role"]
            print(f"  {'OK' if best[0] == 1.0 and w.role == exp['role'] else '~ '} {exp['text']}: read {w.text!r} "
                  f"({best[0]:.2f}), role {w.role}")
        else:
            print(f"  X  {exp['text']}: not found")
    junk = [w.text for j, w in enumerate(writing) if j not in taken]
    extra_boxes = len(boxes) - len(used)
    for t in junk:
        print(f"  !  extra: {t!r}")
    print(f"\nboxes {box_ok}/{len(key['boxes'])} exactly right, axes {n_axes}/{key['axes']}, "
          f"writing {found}/{len(key['writing'])} found ({exact} exact, {role_ok} right role), "
          f"extra items {len(junk) + extra_boxes}")


if __name__ == "__main__":
    main()
