"""把每张卡片以及其最后一行裁出来存图，便于肉眼核对行切分。

用法:
    python tools/snip_cards.py --hwnd 0x00091822
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2

from src.panel import crop, find_cards
from src.win_capture import capture_client_area, enable_dpi_awareness, list_windows, find_windows

OUT = ROOT / "output" / "snips"


def _resolve(args):
    if args.hwnd:
        hwnd = int(args.hwnd, 16) if args.hwnd.lower().startswith("0x") else int(args.hwnd)
        found = [w for w in list_windows(visible_only=False, min_size=0) if w.hwnd == hwnd]
        return found[0] if found else None
    if args.match:
        found = find_windows(args.match, min_size=200)
        return max(found, key=lambda w: w.width * w.height) if found else None
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hwnd")
    parser.add_argument("--match")
    args = parser.parse_args()

    enable_dpi_awareness()
    target = _resolve(args)
    if target is None:
        print("没找到窗口")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    frame = capture_client_area(target.hwnd)
    cards = find_cards(frame)

    for index, card in enumerate(cards):
        img = crop(frame, card.rect, scale=3)
        cv2.imwrite(str(OUT / f"card_{index}_full.png"), img)
        last = card.rows[-1].rect
        cv2.imwrite(str(OUT / f"card_{index}_lastrow.png"), crop(frame, last, scale=4))
        print(f"卡片 {index}: {card.rect}  末行 y={last.y}..{last.y1} h={last.h}")

    print(f"\n已保存到 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
