"""窗口捕获自检工具。

用法:
    python tools/test_capture.py --list                 # 列出所有可见窗口
    python tools/test_capture.py --match 阴阳师          # 对匹配窗口跑全部捕获后端
    python tools/test_capture.py --hwnd 0x00123456      # 直接指定窗口句柄
    python tools/test_capture.py --match 阴阳师 --foreground   # 先置前再截图
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.win_capture import (
    BACKENDS,
    WindowInfo,
    bring_to_front,
    enable_dpi_awareness,
    find_windows,
    list_windows,
)

SHOT_DIR = ROOT / "output" / "screenshots"

# 首选客户区：后台可截、无标题栏、尺寸稳定；其余按可靠性递减
BACKEND_PRIORITY = [
    "client_area",
    "printwindow_full",
    "screen_region",
    "bitblt",
    "printwindow_plain",
]


def _print_windows(windows: list[WindowInfo]) -> None:
    if not windows:
        print("没有找到任何可见窗口。")
        return
    print(f"共 {len(windows)} 个可见窗口：\n")
    for win in sorted(windows, key=lambda w: w.width * w.height, reverse=True):
        print(f"  {win}")
        print()


def _describe(img: np.ndarray) -> tuple[str, str]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean, std = float(gray.mean()), float(gray.std())
    stats = f"mean={mean:6.2f} std={std:6.2f} max={int(gray.max()):3d}"

    if mean < 2.0 and std < 2.0:
        verdict = "全黑 -> 该后端不可用"
    elif std < 3.0:
        verdict = "近乎纯色 -> 可疑"
    else:
        verdict = "有内容 -> 可用"
    return stats, verdict


def _run_backends(target: WindowInfo) -> None:
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    print(f"目标窗口:\n  {target}\n")
    print(f"{'后端':<20} {'尺寸':<14} {'灰度统计':<36} 判定")
    print("-" * 96)

    usable: list[str] = []
    for name, func in BACKENDS.items():
        try:
            img = func(target)
        except Exception as exc:  # 后端失败是预期内的，逐个报告即可
            print(f"{name:<20} {'-':<14} {'-':<36} 异常: {exc}")
            continue

        stats, verdict = _describe(img)
        height, width = img.shape[:2]
        print(f"{name:<20} {f'{width}x{height}':<14} {stats:<36} {verdict}")

        out = SHOT_DIR / f"{stamp}_{name}.png"
        cv2.imwrite(str(out), img)
        if verdict.startswith("有内容"):
            usable.append(name)

    print("-" * 96)
    if usable:
        best = next((n for n in BACKEND_PRIORITY if n in usable), usable[0])
        print(f"可用后端: {', '.join(usable)}")
        print(f"推荐优先使用: {best}")
    else:
        print("没有可用后端。若为 PC 端游，请确认窗口已置前且未被遮挡，再试 --foreground。")
    print(f"\n截图已保存到: {SHOT_DIR}")


def main() -> int:
    parser = argparse.ArgumentParser(description="游戏窗口捕获自检")
    parser.add_argument("--list", action="store_true", help="列出所有可见窗口")
    parser.add_argument("--match", help="按标题/类名关键字匹配窗口")
    parser.add_argument("--hwnd", help="直接指定窗口句柄，支持 0x 前缀")
    parser.add_argument("--foreground", action="store_true", help="截图前把窗口置前")
    parser.add_argument("--min-size", type=int, default=200, help="过滤过小的窗口，默认 200")
    args = parser.parse_args()

    print(f"DPI 感知模式: {enable_dpi_awareness()}\n")

    if args.list or (not args.match and not args.hwnd):
        _print_windows(list_windows(min_size=args.min_size))
        if not args.match and not args.hwnd:
            print("提示: 加 --match 关键字 或 --hwnd 句柄 来执行截图自检。")
        return 0

    if args.hwnd:
        hwnd = int(args.hwnd, 16) if args.hwnd.lower().startswith("0x") else int(args.hwnd)
        matches = [w for w in list_windows(visible_only=False, min_size=0) if w.hwnd == hwnd]
        if not matches:
            print(f"找不到句柄 {args.hwnd} 对应的窗口。")
            return 1
        target = matches[0]
    else:
        matches = find_windows(args.match, min_size=args.min_size)
        if not matches:
            print(f"没有匹配 {args.match!r} 的窗口。用 --list 看看有哪些。")
            return 1
        if len(matches) > 1:
            print(f"匹配到 {len(matches)} 个窗口，使用面积最大的一个：")
            for win in matches:
                print(f"  {win}")
            print()
        target = max(matches, key=lambda w: w.width * w.height)

    if args.foreground:
        bring_to_front(target.hwnd)

    _run_backends(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
