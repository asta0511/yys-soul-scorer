"""面板布局探测：切出每一行文字并导出裁剪图，用于人工核对。

用法:
    python tools/probe_panel.py --hwnd 0x00091822
    python tools/probe_panel.py --hwnd 0x00091822 --threshold 100 --scale 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2

from src.panel import crop, find_rows, panel_roi
from src.win_capture import (
    WindowInfo,
    capture_client_area,
    enable_dpi_awareness,
    find_windows,
    list_windows,
)

OUT_DIR = ROOT / "output" / "panel_probe"


def _resolve_window(args) -> WindowInfo | None:
    if args.hwnd:
        hwnd = int(args.hwnd, 16) if args.hwnd.lower().startswith("0x") else int(args.hwnd)
        found = [w for w in list_windows(visible_only=False, min_size=0) if w.hwnd == hwnd]
        return found[0] if found else None
    if args.match:
        found = find_windows(args.match, min_size=200)
        return max(found, key=lambda w: w.width * w.height) if found else None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="探测御魂强化面板布局")
    parser.add_argument("--hwnd", help="窗口句柄，支持 0x 前缀")
    parser.add_argument("--match", help="按标题关键字匹配窗口")
    parser.add_argument("--threshold", type=int, default=120, help="文字亮度阈值，默认 120")
    parser.add_argument("--scale", type=int, default=4, help="裁剪图放大倍数，默认 4")
    args = parser.parse_args()

    print(f"DPI 感知模式: {enable_dpi_awareness()}\n")

    target = _resolve_window(args)
    if target is None:
        print("没找到目标窗口，用 --list 确认一下。")
        return 1

    frame = capture_client_area(target.hwnd)
    height, width = frame.shape[:2]
    print(f"目标窗口: {target}")
    print(f"客户区尺寸: {width} x {height}\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_DIR / "frame.png"), frame)

    roi = panel_roi(frame)
    print(f"面板 ROI: {roi}")

    roi_crop = crop(frame, roi, scale=2)
    cv2.imwrite(str(OUT_DIR / "panel.png"), roi_crop)

    rows = find_rows(frame, roi=roi, threshold=args.threshold)
    print(f"\n阈值 {args.threshold} 下检测到 {len(rows)} 行：\n")
    print(f"{'行':<4} {'裁剪框':<24} {'行高':<6} {'类型':<8} 相对 y")
    print("-" * 68)

    for index, row in enumerate(rows):
        rect = row.rect
        center_y = rect.y + rect.h / 2
        print(
            f"{index:<4} {str(rect):<24} {rect.h:<6} {row.kind:<8} {center_y / height:.4f}"
        )
        cv2.imwrite(str(OUT_DIR / f"row_{index:02d}.png"), crop(frame, rect, scale=args.scale))

    debug = frame.copy()
    cv2.rectangle(debug, (roi.x, roi.y), (roi.x1, roi.y1), (0, 255, 255), 1)
    for row in rows:
        color = (0, 0, 255) if row.kind == "main" else (0, 255, 0)
        cv2.rectangle(debug, (row.rect.x, row.rect.y), (row.rect.x1, row.rect.y1), color, 1)
    cv2.imwrite(str(OUT_DIR / "debug_bands.png"), debug)

    print(f"\n输出目录: {OUT_DIR}")
    print("  frame.png / panel.png / debug_bands.png / row_NN.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
