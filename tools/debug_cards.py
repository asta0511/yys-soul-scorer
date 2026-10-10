"""批量强化诊断：逐张卡片打印行框坐标与 OCR 原文。

用法:
    python tools/debug_cards.py --hwnd 0x00091822
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ocr import read_region
from src.panel import find_cards
from src.win_capture import capture_client_area, enable_dpi_awareness, find_windows, list_windows


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
    parser = argparse.ArgumentParser(description="批量强化卡片诊断")
    parser.add_argument("--hwnd")
    parser.add_argument("--match")
    parser.add_argument("--threshold", type=int, default=100)
    args = parser.parse_args()

    enable_dpi_awareness()
    target = _resolve(args)
    if target is None:
        print("没找到窗口")
        return 1

    frame = capture_client_area(target.hwnd)
    print(f"客户区 {frame.shape[1]}x{frame.shape[0]}\n")

    cards = find_cards(frame, threshold=args.threshold)
    print(f"检测到 {len(cards)} 张卡片\n")

    for index, card in enumerate(cards):
        lines = read_region(frame, card.rect, list(card.rows))
        print(f"── 卡片 {index} (col={card.column} row={card.row}) 区域 {card.rect}")
        for row, line in zip(card.rows, lines):
            r = row.rect
            print(f"   [{row.kind:<5}] y={r.y:4d}..{r.y1:4d} h={r.h:3d}  {line.text!r}  ({line.score:.3f})")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
