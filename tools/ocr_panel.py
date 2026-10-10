"""面板 OCR 自检：抓窗口 -> 切行 -> 识别 -> 打印文本，导出标注图核对。

用法:
    python tools/ocr_panel.py --hwnd 0x00091822
    python tools/ocr_panel.py --match 鲛人之歌 --scale 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2

from src.ocr import annotate, read_rows
from src.panel import crop, find_rows, panel_roi
from src.win_capture import (
    WindowInfo,
    capture_client_area,
    enable_dpi_awareness,
    find_windows,
    list_windows,
)

OUT_DIR = ROOT / "output" / "ocr_probe"


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
    parser = argparse.ArgumentParser(description="御魂强化面板 OCR 自检")
    parser.add_argument("--hwnd", help="窗口句柄，支持 0x 前缀")
    parser.add_argument("--match", help="按标题关键字匹配窗口")
    parser.add_argument("--threshold", type=int, default=100, help="文字亮度阈值，默认 100")
    parser.add_argument("--scale", type=int, default=2, help="识别前放大倍数，默认 2")
    args = parser.parse_args()

    print(f"DPI 感知模式: {enable_dpi_awareness()}\n")

    target = _resolve_window(args)
    if target is None:
        print("没找到目标窗口。")
        return 1

    frame = capture_client_area(target.hwnd)
    print(f"目标窗口: {target}")
    print(f"客户区尺寸: {frame.shape[1]} x {frame.shape[0]}\n")

    roi = panel_roi(frame)
    rows = find_rows(frame, roi=roi, threshold=args.threshold)
    print(f"检测到 {len(rows)} 行，开始识别（放大 {args.scale}x）...\n")

    started = time.perf_counter()
    lines = read_rows(frame, rows, scale=args.scale)
    elapsed = time.perf_counter() - started

    print(f"{'行':<4} {'类型':<7} {'置信度':<8} 识别文本")
    print("-" * 72)
    for index, line in enumerate(lines):
        print(f"{index:<4} {rows[index].kind:<7} {line.score:<8.3f} {line.text!r}")

    print(f"\n耗时 {elapsed:.2f}s（{elapsed / max(len(lines), 1):.2f}s/行）")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_DIR / "frame.png"), frame)
    cv2.imwrite(str(OUT_DIR / "annotated.png"), annotate(frame, lines))
    for index, line in enumerate(lines):
        cv2.imwrite(str(OUT_DIR / f"row_{index:02d}.png"), crop(frame, line.rect, scale=4))

    print(f"输出目录: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
