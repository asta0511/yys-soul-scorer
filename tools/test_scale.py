"""分辨率自适应自检：把当前帧按不同比例缩放，看卡片/行检测是否仍然稳定。

只验证「检测」环节（ROI + 投影 + 峰值），不跑 OCR——缩放后的截图会糊，OCR 结果
不能代表真机换分辨率的效果。

用法:
    python tools/test_scale.py --hwnd 0x00091822
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2

from src.panel import find_cards, find_rows, panel_roi
from src.win_capture import capture_client_area, enable_dpi_awareness, find_windows, list_windows

SCALES = [0.6, 0.75, 0.9, 1.0, 1.15, 1.3, 1.5]


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
    parser.add_argument("--expect", type=int, default=6, help="期望的卡片数")
    args = parser.parse_args()

    enable_dpi_awareness()
    target = _resolve(args)
    if target is None:
        print("没找到窗口")
        return 1

    frame = capture_client_area(target.hwnd)
    print(f"基准客户区 {frame.shape[1]}x{frame.shape[0]}\n")
    print(f"{'缩放':>6} {'尺寸':>12} {'卡片数':>6} {'行数/卡片':>16}  结果")
    print("-" * 56)

    failed = 0
    for factor in SCALES:
        size = (round(frame.shape[1] * factor), round(frame.shape[0] * factor))
        resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC)

        cards = find_cards(resized)
        counts = ",".join(str(len(c.rows)) for c in cards)
        ok = len(cards) == args.expect
        if not ok:
            failed += 1
        print(f"{factor:>6.2f} {f'{size[0]}x{size[1]}':>12} {len(cards):>6} {counts:>16}  {'ok' if ok else 'FAIL'}")

    # 单面板模式：用同一帧的「详细属性」ROI 走 find_rows，确认行检测也随分辨率稳定
    roi = panel_roi(frame)
    print(f"\n单面板 ROI 行检测（该 ROI 在批量界面里是左侧卡片，仅验证行切分数目）：")
    for factor in SCALES:
        size = (round(frame.shape[1] * factor), round(frame.shape[0] * factor))
        resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC)
        rows = find_rows(resized)
        print(f"  {factor:>5.2f}  {len(rows)} 行")

    print(f"\n{'全部通过' if failed == 0 else f'{failed} 个缩放档位失败'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
