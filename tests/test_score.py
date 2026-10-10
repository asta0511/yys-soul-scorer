"""评分模块的回归测试：拿平安志自己的评分记录当标准答案。

平安志（yys-analysis）把每个御魂的分数和命中方案写在了 analysis_todo 表里。
这里用工程内的 rules/中练度标准.yysrule.json 独立复算一遍，逐条比对，
要求命中方案和分数都完全一致。

数据库不存在时自动跳过（比如换机器跑）。

运行: python tests/test_score.py
"""

from __future__ import annotations

import collections
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.score import (
    AttributeEntry,
    SoulInfo,
    entries_from_parsed,
    load_rules,
    main_attribute,
    score_soul,
    slot_candidates,
)

DB_PATH = Path(r"C:\Users\Administrator\AppData\Roaming\com.yys.analysis\current-analysis.sqlite3")


def _load_db() -> tuple[dict, dict, dict, list] | None:
    if not DB_PATH.exists():
        return None
    con = sqlite3.connect("file:%s?immutable=1" % DB_PATH, uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    keymap = {}
    cur.execute("select soul_key, snapshot_id, soul_internal_id from inventory_item")
    for r in cur.fetchall():
        keymap[r["soul_key"]] = (r["snapshot_id"], r["soul_internal_id"])

    souls = {}
    cur.execute(
        "select snapshot_id, internal_id, set_id, slot, quality, main_attr_type from snapshot_soul"
    )
    for r in cur.fetchall():
        souls[(r["snapshot_id"], r["internal_id"])] = dict(r)

    attrs: dict[tuple, list] = collections.defaultdict(list)
    cur.execute(
        "select snapshot_id, soul_internal_id, attribute_index, attribute_type, enhancement_count "
        "from soul_attribute order by attribute_index"
    )
    for r in cur.fetchall():
        attrs[(r["snapshot_id"], r["soul_internal_id"])].append(dict(r))

    todo = []
    cur.execute("select soul_key, standard_score, detail_json from analysis_todo")
    for r in cur.fetchall():
        todo.append(dict(r))
    return keymap, souls, attrs, todo


def _check_pure(rules: list) -> int:
    """不依赖数据库的检查：号位推断、属性行映射、套装未知分支。"""
    failed = 0

    # 号位推断：主属性唯一确定号位的应当只给一个候选
    assert slot_candidates("attack_flat") == (1,)
    assert slot_candidates("crit_damage") == (6,)
    assert slot_candidates("attack_rate") == (2, 4, 6)

    # 面板行 -> 评分条目：主属性跳过，徽章按「原始行号」对上，不能按解析后的顺序
    indexed = [
        (0, {"kind": "main", "name": "防御"}),
        (1, {"kind": "sub", "name": "效果抵抗"}),
        (2, {"kind": "sub", "name": "生命加成"}),
        (3, {"kind": "innate", "name": "防御加成"}),
    ]
    entries = entries_from_parsed(indexed, {1: 3})
    assert [(e.attr_id, e.enhancement_count) for e in entries] == [
        ("effect_resist", 3),
        ("hp_rate", 0),
        ("defense_rate", 0),
    ], entries
    assert main_attribute(indexed) == "defense_flat"

    # 套装未知时要跳过套装条件取最高分，并标 set_verified=False
    loose = score_soul(
        SoulInfo(attributes=entries, slot=3, quality=6, main_attr="defense_flat"), rules
    )
    if loose is None or loose.slot != 3 or loose.set_verified is not False:
        print("[FAIL] 套装未知分支不对:", loose)
        failed += 1

    # 给出套装后按套装过滤，set_verified 转 True
    set_ids = sorted({sid for rule in rules for sid in rule.set_ids})
    if not set_ids:
        print("[FAIL] 规则里没有套装列表")
        failed += 1
    else:
        strict = score_soul(
            SoulInfo(attributes=entries, set_id=set_ids[0], slot=3, quality=6, main_attr="defense_flat"),
            rules,
        )
        if strict is not None and strict.set_verified is not True:
            print("[FAIL] 指定套装后 set_verified 应当为 True:", strict)
            failed += 1
    return failed


def main() -> int:
    rules = load_rules()
    print(f"载入方案 {len(rules)} 条")
    failed = _check_pure(rules)

    loaded = _load_db()
    if loaded is None:
        print("没找到平安志数据库，跳过分数比对。")
        return 1 if failed else 0
    keymap, souls, attrs, todo = loaded

    checked = ok_score = ok_rule = 0
    bad: list[tuple] = []
    for item in todo:
        loc = keymap.get(item["soul_key"])
        if not loc:
            continue
        soul = souls.get(loc)
        if not soul:
            continue
        entries = [
            AttributeEntry(attr_id=a["attribute_type"], enhancement_count=a["enhancement_count"] or 0)
            for a in attrs.get(loc, [])
        ]
        info = SoulInfo(
            attributes=entries,
            set_id=soul["set_id"],
            slot=soul["slot"],
            quality=soul["quality"],
            main_attr=soul["main_attr_type"],
        )
        result = score_soul(info, rules)
        if result is None:
            continue

        expected = (json.loads(item["detail_json"]).get("standardScore") or {})
        checked += 1
        if abs(result.score - item["standard_score"]) < 1e-6:
            ok_score += 1
        else:
            bad.append((item["standard_score"], result.score, result.rule_id, expected.get("matchedRuleId")))
        if result.rule_id == expected.get("matchedRuleId"):
            ok_rule += 1

    print(f"比对 {checked} 条")
    print(f"  分数一致 {ok_score} ({100.0 * ok_score / max(checked, 1):.2f}%)")
    print(f"  方案一致 {ok_rule} ({100.0 * ok_rule / max(checked, 1):.2f}%)")
    if bad:
        print("  前几条不一致:", bad[:5])

    if checked and ok_score != checked:
        print("[FAIL] 分数没有完全对上")
        failed += 1
    if checked and ok_rule != checked:
        print("[FAIL] 方案没有完全对上")
        failed += 1
    if not checked:
        print("[FAIL] 一条都没比对到")
        failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
