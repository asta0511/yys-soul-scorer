"""打印某一列在指定 y 范围内的逐行亮度统计，用于判断边框行与文字行的区别。

用法:
    python tools/profile_column.py --hwnd 0x00091822 --box 555,620,831,715
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.panel import Rect
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--hwnd")
    parser.add_argument("--match")
    parser.add_argument("--box", required=True, help="x0,y0,x1,y1")
    parser.add_argument("--threshold", type=int, default=100)
    parser.add_argument("--bands", action="store_true", help="只打印行段峰值汇总")
    args = parser.parse_args()

    enable_dpi_awareness()
    target = _resolve(args)
    if target is None:
        print("没找到窗口")
        return 1

    x0, y0, x1, y1 = (int(v) for v in args.box.split(","))
    frame = capture_client_area(target.hwnd)
    region = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    mask = (gray >= args.threshold).astype(np.uint8)
    counts = mask.sum(axis=1)
    coverage = (mask.sum(axis=0) > 0).sum() / mask.shape[1]

    print(f"区域宽 {mask.shape[1]}，阈值 {args.threshold}，横向覆盖率 {coverage:.2f}\n")

    if args.bands:
        from src.panel import _bands

        width = mask.shape[1]
        bands = _bands(counts, min_len=6, min_pixels=3, merge_gap=3)
        print(f"{'band':>16} {'高':>4} {'峰值':>5} {'占比':>6} {'总亮':>6}")
        for start, end in bands:
            peak = int(counts[start:end].max())
            total = int(counts[start:end].sum())
            print(f"{y0 + start:>6}..{y0 + end:<6} {end - start:>4} {peak:>5} {peak / width:>6.2f} {total:>6}")
        return 0

    print(f"{'y':>5} {'亮像素':>6}  直方")
    for offset, count in enumerate(counts):
        y = y0 + offset
        bar = "#" * int(count / 2)
        print(f"{y:>5} {count:>6}  {bar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
