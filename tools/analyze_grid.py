"""批量强化网格分析：找出御魂卡片的列/行边界。

批量强化界面是 N 列 × M 行的卡片网格，每个卡片内部结构与单面板一致。
本工具对整幅客户区做行/列投影，把「有文字的带」和「卡片之间的空隙」分开，
用来标定列 x 区间、行 y 区间。

用法:
    python tools/analyze_grid.py --hwnd 0x00091822
    python tools/analyze_grid.py --image output/screenshots/batch_now.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.win_capture import enable_dpi_awareness, list_windows

OUT_DIR = ROOT / "output" / "grid_probe"


def _segments(active: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    segs: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(active):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if index - start >= min_len:
                segs.append((start, index))
            start = None
    if start is not None and len(active) - start >= min_len:
        segs.append((start, len(active)))
    return segs


def _report(name: str, counts: np.ndarray, offset: int, min_len: int, min_pixels: int) -> None:
    active = counts >= min_pixels
    segs = _segments(active, min_len)

    print(f"\n===== {name} =====")
    print(f"{'起(全图)':<10} {'止(全图)':<10} {'长度':<7} {'峰值':<7} 与前段间隔")
    print("-" * 58)
    previous_end = None
    for start, end in segs:
        gap = "-" if previous_end is None else str(start - previous_end)
        print(f"{start + offset:<10} {end + offset:<10} {end - start:<7} "
              f"{int(counts[start:end].max()):<7} {gap}")
        previous_end = end
    print(f"共 {len(segs)} 段")


def main() -> int:
    parser = argparse.ArgumentParser(description="批量强化网格布局分析")
    parser.add_argument("--hwnd", help="窗口句柄，支持 0x 前缀")
    parser.add_argument("--image", help="直接分析已有图片")
    parser.add_argument("--threshold", type=int, default=100, help="文字亮度阈值，默认 100")
    parser.add_argument("--min-pixels", type=int, default=2, help="判定为「有文字」的最小像素数")
    parser.add_argument("--min-len", type=int, default=4, help="最小段长度")
    args = parser.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"读不到图片: {args.image}")
            return 1
    else:
        enable_dpi_awareness()
        if not args.hwnd:
            print("需要 --hwnd 或 --image")
            return 1
        hwnd = int(args.hwnd, 16) if args.hwnd.lower().startswith("0x") else int(args.hwnd)
        found = [w for w in list_windows(visible_only=False, min_size=0) if w.hwnd == hwnd]
        if not found:
            print(f"找不到窗口 {args.hwnd}")
            return 1
        from src.win_capture import capture_client_area

        frame = capture_client_area(hwnd)

    height, width = frame.shape[:2]
    print(f"图像尺寸: {width} x {height}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bright = (gray >= args.threshold).astype(np.uint8)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_DIR / "frame.png"), frame)
    cv2.imwrite(str(OUT_DIR / "bright.png"), (bright * 255))

    _report("列投影（整幅，找卡片列）", bright.sum(axis=0), 0, args.min_len, args.min_pixels)
    _report("行投影（整幅，找卡片行）", bright.sum(axis=1), 0, args.min_len, args.min_pixels)

    print(f"\n输出: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
