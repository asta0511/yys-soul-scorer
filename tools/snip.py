"""从游戏窗口客户区裁一块指定矩形并放大保存，用于人工核对像素坐标。

用法:
    python tools/snip.py --hwnd 0x00091822 --rect 420,160,560,350 --scale 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2

from src.win_capture import (
    WindowInfo,
    capture_client_area,
    enable_dpi_awareness,
    find_windows,
    list_windows,
)

OUT_DIR = ROOT / "output" / "snips"


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
    parser = argparse.ArgumentParser(description="裁剪客户区指定区域")
    parser.add_argument("--hwnd", help="窗口句柄，支持 0x 前缀")
    parser.add_argument("--match", help="按标题关键字匹配窗口")
    parser.add_argument("--rect", required=True, help="x0,y0,x1,y1")
    parser.add_argument("--scale", type=int, default=3, help="放大倍数，默认 3")
    parser.add_argument("--name", default="snip", help="输出文件名（不含扩展名）")
    args = parser.parse_args()

    enable_dpi_awareness()

    target = _resolve_window(args)
    if target is None:
        print("没找到目标窗口。")
        return 1

    x0, y0, x1, y1 = (int(v) for v in args.rect.split(","))
    frame = capture_client_area(target.hwnd)
    height, width = frame.shape[:2]
    x0, x1 = max(x0, 0), min(x1, width)
    y0, y1 = max(y0, 0), min(y1, height)

    crop = frame[y0:y1, x0:x1]
    if args.scale != 1:
        crop = cv2.resize(crop, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_NEAREST)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.name}.png"
    cv2.imwrite(str(out), crop)

    print(f"客户区 {width}x{height}  裁剪 {x0},{y0}..{x1},{y1}  ->  {crop.shape[1]}x{crop.shape[0]}")
    print(f"已保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
