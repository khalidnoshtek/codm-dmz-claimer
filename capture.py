#!/usr/bin/env python3
"""Capture helper for building the templates/ library.

Usage:
    # Take a full screenshot of whatever the AVD is showing right now
    python capture.py screen --out logs/now.png

    # Capture a rectangle as a template (top-left x,y to bottom-right X,Y)
    python capture.py crop --src logs/now.png --rect 480 720 720 800 --out templates/04_claim_button.png

    # Verify a template matches the current screen
    python capture.py match --template templates/04_claim_button.png

The intended workflow:
1) Navigate the emulator (manually) to the screen you want to capture.
2) `python capture.py screen --out logs/screen-<name>.png` to grab the full frame.
3) Open that PNG in Preview / any image editor, note the (x,y,X,Y) of the button you care about.
4) `python capture.py crop --src logs/screen-<name>.png --rect x y X Y --out templates/<NN_name>.png`
5) `python capture.py match --template templates/<NN_name>.png` to confirm it still matches.

Capture each button referenced in lib/flow.py's DEFAULT_STEPS this way.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import yaml

from lib.adb import AdbDevice
from lib.vision import find_template, find_all

ROOT = Path(__file__).resolve().parent


def _device() -> AdbDevice:
    """Get the locked CODM AVD. The picker hard-locks to AVD `LOCKED_AVD`
    (see lib/adb.py) — we never even pass config here, to make the lock
    impossible to subvert from this entry point."""
    return AdbDevice.auto()


def cmd_screen(args) -> int:
    device = _device()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    device.screencap_to(out)
    print(f"Saved {out} (device={device.serial})")
    return 0


def cmd_shot(args) -> int:
    """Take an AVD screenshot, save it at the target template path, then open
    it in macOS Preview so the user can crop + save in place.

    Recommended workflow per template:
      .venv/bin/python capture.py shot --name 04_lst_hunt_tab
      # Preview opens the file. Drag-select the button, Cmd-K to crop,
      # Cmd-S to overwrite the same file. Done.
      .venv/bin/python capture.py match --template templates/04_lst_hunt_tab.png
    """
    import subprocess as _sub
    device = _device()
    name = args.name
    if not name:
        name = input("Template name (e.g. 04_lst_hunt_tab): ").strip()
    if not name:
        print("Aborted: no name given.", file=sys.stderr)
        return 2
    if not name.endswith(".png"):
        name += ".png"
    out_path = (ROOT / "templates" / name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device.screencap_to(out_path)
    print(f"Saved full AVD screenshot to {out_path}")
    print(f"  Opening in Preview — crop to the button area (drag-select + Cmd-K), then Cmd-S.")
    print(f"  After saving, verify:  .venv/bin/python capture.py match --template {out_path.relative_to(ROOT)}")
    try:
        _sub.run(["open", "-a", "Preview", str(out_path)], check=False)
    except Exception as e:
        print(f"  (could not auto-open Preview: {e}; open the file manually)")
    return 0


def cmd_crop(args) -> int:
    src = Path(args.src)
    img = cv2.imread(str(src))
    if img is None:
        print(f"Could not read {src}", file=sys.stderr)
        return 2
    x1, y1, x2, y2 = args.rect
    if x2 <= x1 or y2 <= y1:
        print("Rect must be x1 y1 x2 y2 with x2>x1 and y2>y1.", file=sys.stderr)
        return 2
    crop = img[y1:y2, x1:x2]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), crop)
    print(f"Saved {out} ({x2-x1}x{y2-y1})")
    return 0


def cmd_interactive(args) -> int:
    """Live screenshot -> drag a rectangle -> save as a template.

    Workflow per template:
      1) On the AVD, navigate to the screen that contains the button.
      2) Run:  python capture.py interactive --name 04_claim_button
      3) An OpenCV window opens showing what the AVD sees right now.
      4) Click-drag a rectangle around the button.
         - ENTER  -> save crop to templates/<name>.png, verify the match.
         - r      -> grab a fresh screenshot (use if the AVD changed).
         - ESC/q  -> cancel without saving.
    """
    import cv2 as _cv

    device = _device()
    templates_dir = ROOT / "templates"
    templates_dir.mkdir(exist_ok=True)
    name = args.name
    if not name:
        name = input("Template name (e.g. 04_claim_button): ").strip()
    if not name:
        print("Aborted: no name given.", file=sys.stderr)
        return 2
    if not name.endswith(".png"):
        name += ".png"
    out_path = templates_dir / name

    state = {"screen": device.screencap(), "drag": None, "rect": None, "preview": None}

    def render():
        img = state["screen"].copy()
        if state["drag"] is not None:
            x1, y1, x2, y2 = state["drag"]
            _cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        elif state["rect"] is not None:
            x1, y1, x2, y2 = state["rect"]
            _cv.rectangle(img, (x1, y1), (x2, y2), (0, 200, 255), 3)
        # Down-scale for display so 1080x2400 fits on a Mac screen
        h, w = img.shape[:2]
        scale = min(1.0, 900 / w, 1000 / h)
        if scale < 1.0:
            disp = _cv.resize(img, (int(w * scale), int(h * scale)))
        else:
            disp = img
        state["preview"] = disp
        state["scale"] = scale
        _cv.imshow(WIN, disp)

    def on_mouse(event, x, y, flags, _userdata):
        scale = state["scale"]
        # Map preview coords back to native screen coords
        sx, sy = int(x / scale), int(y / scale)
        if event == _cv.EVENT_LBUTTONDOWN:
            state["drag"] = (sx, sy, sx, sy)
            state["rect"] = None
            render()
        elif event == _cv.EVENT_MOUSEMOVE and state["drag"] is not None and (flags & _cv.EVENT_FLAG_LBUTTON):
            x1, y1, _, _ = state["drag"]
            state["drag"] = (x1, y1, sx, sy)
            render()
        elif event == _cv.EVENT_LBUTTONUP and state["drag"] is not None:
            x1, y1, x2, y2 = state["drag"]
            state["drag"] = None
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            if x2 - x1 < 10 or y2 - y1 < 10:
                state["rect"] = None
                print("(rectangle too small — try again)")
            else:
                state["rect"] = (x1, y1, x2, y2)
                print(f"selection: {x1},{y1} -> {x2},{y2}  ({x2-x1}x{y2-y1})")
            render()

    WIN = f"capture: {name}"
    _cv.namedWindow(WIN)
    _cv.setMouseCallback(WIN, on_mouse)
    print("Instructions: drag a rectangle.  ENTER=save  r=refresh screen  ESC/q=cancel")
    render()

    while True:
        key = _cv.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            print("cancelled.")
            _cv.destroyAllWindows()
            return 1
        if key == ord("r"):
            print("refreshing screenshot...")
            state["screen"] = device.screencap()
            state["rect"] = None
            render()
        if key in (10, 13):  # enter
            if state["rect"] is None:
                print("(nothing selected yet)")
                continue
            x1, y1, x2, y2 = state["rect"]
            crop = state["screen"][y1:y2, x1:x2]
            _cv.imwrite(str(out_path), crop)
            print(f"saved {out_path}  ({x2-x1}x{y2-y1})")
            # Verify it matches right back where we cut it from
            from lib.vision import find_template
            m = find_template(state["screen"], out_path, threshold=0.7)
            if m:
                print(f"verify: matched at ({m.x},{m.y}) score={m.score:.3f}")
            else:
                print("verify: no match >= 0.70 — crop may be too tight or include too much background.")
            _cv.destroyAllWindows()
            return 0


def cmd_match(args) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = _device()
    screen = device.screencap()
    tpl_path = Path(args.template)
    threshold = float(args.threshold)
    if args.all:
        hits = find_all(screen, tpl_path, threshold=threshold)
        if not hits:
            print(f"No matches >= {threshold} for {tpl_path.name}")
            return 1
        for i, h in enumerate(hits):
            print(f"[{i}] center=({h.x},{h.y}) score={h.score:.3f}")
    else:
        m = find_template(screen, tpl_path, threshold=threshold)
        if not m:
            print(f"No match >= {threshold} for {tpl_path.name}")
            return 1
        print(f"center=({m.x},{m.y}) score={m.score:.3f} size={m.w}x{m.h}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Template capture / verify helper.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_scr = sub.add_parser("screen", help="Save the current AVD screen to a PNG.")
    p_scr.add_argument("--out", default=str(ROOT / "logs" / "screen.png"))
    p_scr.set_defaults(func=cmd_screen)

    p_shot = sub.add_parser(
        "shot",
        help="Screencap AVD, save to templates/<name>.png, open in Preview to crop in place.",
    )
    p_shot.add_argument("--name", help="Template filename (without templates/ prefix).")
    p_shot.set_defaults(func=cmd_shot)

    p_crop = sub.add_parser("crop", help="Crop a rectangle out of a saved screenshot into templates/.")
    p_crop.add_argument("--src", required=True)
    p_crop.add_argument("--rect", nargs=4, type=int, required=True, metavar=("x1", "y1", "x2", "y2"))
    p_crop.add_argument("--out", required=True)
    p_crop.set_defaults(func=cmd_crop)

    p_int = sub.add_parser("interactive", help="Drag-to-crop a region from a live screenshot and save as a template.")
    p_int.add_argument("--name", help="Template filename (without templates/ prefix).")
    p_int.set_defaults(func=cmd_interactive)

    p_match = sub.add_parser("match", help="Test a template against the current screen.")
    p_match.add_argument("--template", required=True)
    p_match.add_argument("--threshold", default=0.82)
    p_match.add_argument("--all", action="store_true", help="Find every occurrence, not just the best.")
    p_match.set_defaults(func=cmd_match)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
