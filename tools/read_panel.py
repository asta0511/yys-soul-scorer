"""完整流程：抓窗口 -> 定位卡片/面板 -> 切行 -> OCR -> 解析 -> 读强化次数 -> 识别套装 -> 评分 -> 输出。

识别本身在 src/pipeline.py（图形界面 tools/gui.py 也调它），这里只负责命令行参数、
把结果打印成表格、以及写 JSON/CSV/标注图。

两种界面都认（判别在 src/pipeline.py，详情页优先）：

  * 御魂详情页——仓库/列表右侧弹出的面板，识别选中的那一件。号位读「壹号位」
    文本，套装名读面板文字，徽章在属性值右边。
  * 御魂批量强化的结果页。一屏 N 张卡片就用网格切（find_cards），只剩一张时同样
    按卡片走，这样每张卡片右侧的强化次数徽章都能读到。套装靠卡片头部左侧的圆形
    图标识别（src/seticon.py，模板在 data/soul_icons/）。

套装识别不出时 set_id 传 None，评分取「所有套装方案里最高的那个」，结果里
set_verified=False，同时列出备选方案。

批量页的号位优先取图标背后的角标（src/seticon.slot_from_icon）；角标判不出或与
主属性矛盾时退回「主属性推断」，比例型主属性（攻击/防御/生命加成）就只能落在
2/4/6 三个候选里取最高分，结果里标 slot_inferred。

用法:
    python tools/read_panel.py --hwnd 0x00091822
    python tools/read_panel.py --match 鲛人之歌 --json out.json --csv out.csv
    python tools/read_panel.py --image output/screenshots/batch_now.png   # 离线跑截图
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.ocr import annotate
from src.parse import format_value, unparsed
from src.pipeline import (
    MODE_BATCH,
    MODE_SINGLE_CARD,
    MODE_SOUL_DETAIL,
    AnalysisError,
    SoulReport,
    analyze_frame,
)
from src.score import load_rules
from src.win_capture import (
    WindowInfo,
    capture_client_area,
    enable_dpi_awareness,
    find_windows,
    list_windows,
)

OUT_DIR = ROOT / "output" / "panel_read"

KIND_LABEL = {"main": "主属性", "sub": "副属性", "innate": "固有"}


def _resolve_window(args) -> WindowInfo | None:
    if args.hwnd:
        hwnd = int(args.hwnd, 16) if args.hwnd.lower().startswith("0x") else int(args.hwnd)
        found = [w for w in list_windows(visible_only=False, min_size=0) if w.hwnd == hwnd]
        return found[0] if found else None
    if args.match:
        found = find_windows(args.match, min_size=200)
        return max(found, key=lambda w: w.width * w.height) if found else None
    return None


def _load_frame(args) -> tuple[np.ndarray | None, str]:
    """返回 (画面, 标题)。--image 走离线截图，否则抓窗口客户区。"""
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"读不出图片: {args.image}")
        return frame, Path(args.image).name

    target = _resolve_window(args)
    if target is None:
        print("没找到目标窗口。")
        return None, ""
    return capture_client_area(target.hwnd), target.title


def _print_soul(report: SoulReport) -> None:
    for warning in report.warnings:
        print(f"[警告] {warning}")

    print(f"\n── 御魂 {report.index + 1} ──────────────────────────────────")
    set_match = report.set_match
    if set_match is None:
        print("套装: 未识别出")
    elif set_match.source == "text":
        print(f"套装: {set_match.name}（面板文字）")
    else:
        print(f"套装: {set_match.name}（分数 {set_match.score:.3f}，领先 {set_match.margin:.3f}）")
    print(f"{'类型':<6} {'属性名':<10} {'旧值':<10} {'最终值':<10} {'强化':<6} 置信度")
    print("-" * 62)
    for row_index, item in report.indexed:
        if report.badges:
            mark = str(report.badges.get(row_index, 0))
        else:
            mark = "是" if item["upgraded"] else "否"
        print(
            f"{KIND_LABEL.get(item['kind'], item['kind']):<6} {item['name']:<10} "
            f"{format_value(item['old'], item['unit']):<10} {format_value(item['final'], item['unit']):<10} "
            f"{mark:<6} {item['score']:.3f}"
        )
    for line in unparsed(report.lines, kinds=report.kinds):
        print(f"[警告] 没解析出来的行: {line.text!r}（置信度 {line.score:.3f}）")

    result = report.result
    if result is None:
        print("  评分: 没有命中的方案")
        return
    flag = "" if result.set_verified else "  [套装未识别 · 乐观上界]"
    slot = f"{result.slot}号位" if result.slot else "号位未知"
    if result.slot_inferred:
        slot += "(推断)"
    print(f"  评分 {result.score:.0f} · 方案 {result.rule_name} · {slot} · 建议 {result.verdict}{flag}")
    if not result.set_verified and len(result.candidates) > 1:
        alt = "、".join(f"{name}({score:.0f}/{verdict})" for name, score, verdict in result.candidates[1:4])
        print(f"  备选方案: {alt}")


def main() -> int:
    parser = argparse.ArgumentParser(description="读取御魂强化面板属性并评分")
    parser.add_argument("--hwnd", help="窗口句柄，支持 0x 前缀")
    parser.add_argument("--match", help="按标题关键字匹配窗口")
    parser.add_argument("--image", help="离线测试：直接读一张截图，不抓窗口")
    parser.add_argument("--threshold", type=int, default=100, help="文字亮度阈值，默认 100")
    parser.add_argument("--scale", type=int, default=2, help="识别前放大倍数，默认 2")
    parser.add_argument("--json", help="JSON 输出路径")
    parser.add_argument("--csv", help="CSV 输出路径")
    parser.add_argument("--no-save", action="store_true", help="不保存调试图")
    args = parser.parse_args()

    enable_dpi_awareness()

    frame, title = _load_frame(args)
    if frame is None:
        return 1
    print(f"窗口: {title}  客户区 {frame.shape[1]}x{frame.shape[0]}")

    rules = load_rules()

    started = time.perf_counter()
    try:
        analysis = analyze_frame(frame, rules, threshold=args.threshold, scale=args.scale)
    except AnalysisError as exc:
        print(str(exc))
        return 1
    elapsed = time.perf_counter() - started

    if analysis.mode == MODE_SOUL_DETAIL:
        print("御魂详情面板：选中的这一件")
    elif analysis.mode == MODE_BATCH:
        print(f"批量强化：{len(analysis.souls)} 张卡片")
    elif analysis.mode == MODE_SINGLE_CARD:
        print("单个御魂面板（按卡片处理，可读强化次数）")
    else:
        print("单个御魂面板（没检测到卡片，读不到强化次数）")

    for report in analysis.souls:
        _print_soul(report)

    print(f"\n共 {len(analysis.souls)} 个御魂，耗时 {elapsed:.2f}s")

    if not args.no_save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        all_lines = [line for report in analysis.souls for line in report.lines]
        cv2.imwrite(str(OUT_DIR / "annotated.png"), annotate(frame, all_lines))

    payload = analysis.to_dict(title, time.strftime("%Y-%m-%d %H:%M:%S"))

    if args.json:
        Path(args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"JSON 已写入: {args.json}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "soul", "kind", "name", "old", "final", "unit", "upgraded",
                "ocr_score", "enhance_count", "soul_score", "rule", "verdict",
            ])
            for soul in analysis.souls:
                score = soul.result.to_dict() if soul.result else {}
                for row_index, item in soul.indexed:
                    writer.writerow([
                        soul.index + 1, item["kind"], item["name"], item["old"],
                        item["final"], item["unit"], int(item["upgraded"]),
                        f"{item['score']:.3f}", soul.badges.get(row_index, 0),
                        score.get("score"), score.get("rule_name"), score.get("verdict"),
                    ])
        print(f"CSV 已写入: {args.csv}")

    if not args.no_save:
        print(f"标注图: {OUT_DIR / 'annotated.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
