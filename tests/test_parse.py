"""parse 解析的回归测试，重点覆盖「蓝色双箭头被 OCR 漏掉」的情况。

运行: python tests/test_parse.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ocr import OcrLine
from src.panel import Rect
from src.parse import parse_line

# (OCR 原文, 期望属性名, 期望旧值, 期望最终值)
CASES = [
    # 箭头被读成 ">" / "》" / ")" / "《" —— 正常
    ("防御14.00 > 104.00", "防御", 14.0, 104.0),
    ("效果抵抗3.62% 》 14.64%", "效果抵抗", 3.62, 14.64),
    ("攻击22.32 ) 74.77", "攻击", 22.32, 74.77),
    ("速度2.50《5.49", "速度", 2.5, 5.49),
    # 箭头整个漏掉，两个数直接相邻（此前会被并成一个数）
    ("防御14.00  104.00", "防御", 14.0, 104.0),
    ("生命98.99 192.41", "生命", 98.99, 192.41),
    ("生命342.00 >2052.00", "生命", 342.0, 2052.0),
    # 单值（本次没被强化）
    ("暴击伤害3.47%", "暴击伤害", 3.47, 3.47),
    ("速度2.59", "速度", 2.59, 2.59),
    # 固有属性前缀
    ("固有属性：防御加成16.00%", "防御加成", 16.0, 16.0),
    # 批量强化里副属性左侧的绿色「新」标签
    ("新暴击0.00% > 2.80%", "暴击", 0.0, 2.8),
]


def main() -> int:
    failed = 0
    for raw, name, old, final in CASES:
        line = OcrLine(rect=Rect(0, 0, 10, 10), text=raw, score=0.9)
        parsed = parse_line(line, "sub")
        if parsed is None:
            print(f"[FAIL] {raw!r} 没解析出结果")
            failed += 1
            continue
        ok = parsed.name == name and parsed.old == old and parsed.final == final
        flag = "ok  " if ok else "FAIL"
        print(f"[{flag}] {raw!r:<28} -> {parsed.name} {parsed.old} -> {parsed.final}")
        if not ok:
            print(f"        期望 {name} {old} -> {final}")
            failed += 1

    print(f"\n{len(CASES) - failed}/{len(CASES)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
