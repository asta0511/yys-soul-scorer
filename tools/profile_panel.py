"""行投影剖面诊断：打印面板区域内每一行的亮像素统计，用于标定 ROI 与阈值。

用法:
    python tools/profile_panel.py --hwnd 0x00091822
    python tools/profile_panel.py --hwnd 0x00091822 --threshold 100 --min-count 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.win_capture import (
    WindowInfo,
    capture_client_area,
    enable_dpi_awareness,
    find_windows,
    list_windows,
)

# 故意开得比面板大，好看出面板上下边界在哪
SCAN_ROI = (0.150, 0.070, 0.390, 0.800)


def _resolve_window(args) -> WindowInfo | None:
    if args.hwnd:
        hwnd = int(args.hwnd, 16) if args.hwnd.lower().startswith("0x") else int(args.hwnd)
        found = [w for w in list_windows(visible_only=False, min_size=0) if w.hwnd == hwnd]
        return found[0] if found else None
    if args.match:
        found = find_windows(args.match, min_size=200)
        return max(found, key=lambda w: w.width * w.height) if found else None
    return None


def _column_profile(bright, rows_range: str, x0: int, y0: int, roi_w: int) -> int:
    start_y, end_y = (int(v) for v in rows_range.split(".."))
    sub = bright[start_y - y0 : end_y - y0]
    counts = sub.sum(axis=0)

    print(f"列投影统计范围: y={start_y}..{end_y}\n")
    print(f"{'x(全图)':<12} {'x(全图)尾':<12} {'宽':<6} {'峰count':<9} 说明")
    print("-" * 60)

    segments: list[list[int]] = []
    start: int | None = None
    for index, value in enumerate(counts):
        if value > 0 and start is None:
            start = index
        elif value == 0 and start is not None:
            segments.append([start, index])
            start = None
    if start is not None:
        segments.append([start, len(counts)])

    previous_end = None
    for seg_start, seg_end in segments:
        gap = "" if previous_end is None else f"与前段间隔 {seg_start - previous_end}"
        print(
            f"{seg_start + x0:<12} {seg_end + x0:<12} {seg_end - seg_start:<6} "
            f"{int(counts[seg_start:seg_end].max()):<9} {gap}"
        )
        previous_end = seg_end

    print("-" * 60)
    print(f"共 {len(segments)} 段（段之间的空白就是列间距）")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="行投影剖面诊断")
    parser.add_argument("--hwnd", help="窗口句柄，支持 0x 前缀")
    parser.add_argument("--match", help="按标题关键字匹配窗口")
    parser.add_argument("--threshold", type=int, default=100)
    parser.add_argument("--min-count", type=int, default=2, help="判定该行有内容的亮像素下限")
    parser.add_argument("--columns", action="store_true", help="改为输出列投影（用于定位徽章）")
    parser.add_argument("--rows-range", default="160..400", help="列投影时统计的行范围，默认 160..400")
    args = parser.parse_args()

    enable_dpi_awareness()

    target = _resolve_window(args)
    if target is None:
        print("没找到目标窗口。")
        return 1

    frame = capture_client_area(target.hwnd)
    height, width = frame.shape[:2]
    print(f"客户区: {width} x {height}\n")

    x0 = int(width * SCAN_ROI[0])
    x1 = int(width * SCAN_ROI[2])
    y0 = int(height * SCAN_ROI[1])
    y1 = int(height * SCAN_ROI[3])
    roi = frame[y0:y1, x0:x1]
    roi_w = x1 - x0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bright = (gray >= args.threshold).astype(np.uint8)
    counts = bright.sum(axis=1)

    print(f"扫描区: x={x0}..{x1} (宽 {roi_w})  y={y0}..{y1}  阈值={args.threshold}\n")

    if args.columns:
        return _column_profile(bright, args.rows_range, x0, y0, roi_w)

    print(f"{'y(全图)':<9} {'高':<5} {'均count':<9} {'峰count':<9} {'峰/宽':<8} 判定")
    print("-" * 66)

    bands: list[list[int]] = []
    start: int | None = None
    for index, value in enumerate(counts):
        if value >= args.min_count and start is None:
            start = index
        elif value < args.min_count and start is not None:
            bands.append([start, index])
            start = None
    if start is not None:
        bands.append([start, len(counts)])

    for band_start, band_end in bands:
        segment = counts[band_start:band_end]
        peak = int(segment.max())
        mean = float(segment.mean())
        ratio = peak / roi_w
        if ratio > 0.7:
            verdict = "整行亮线 -> 面板边框"
        elif band_end - band_start >= 60:
            verdict = "很高 -> 图标区"
        else:
            verdict = "文字行"
        print(
            f"{band_start + y0:<9} {band_end - band_start:<5} {mean:<9.1f} "
            f"{peak:<9} {ratio:<8.2f} {verdict}"
        )

    print("-" * 66)
    print(f"共 {len(bands)} 段")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
