"""按「中练度标准.yysrule.json」给御魂打分。

评分公式（与平安志 builtin.owner.default-manual 方案完全一致，用它的 5808 条
评分记录回归验证过：命中方案 100%、分数 100%）：

    分数 = Σ 命中的核心属性 (10 + 强化次数 × 10)
         + Σ 命中的一般属性 (5  + 强化次数 × 5)

「命中」是逐条算的：副属性和固有属性各算一条，同名属性可以叠加；主属性不参与计分。

方案匹配只看四项——星级、号位、号位主属性配对、套装。等级、初始腿数、
必备副属性这些字段不参与评分匹配，这是实测结论：平安志算 standardScore 时
也不看它们（胚子准入是另一套判定，不走这里）。

留/观察/弃直接取命中方案自己的阈值（keepScore / observeScore / discardScore）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.paths import resource_root

DEFAULT_RULE_PATH = resource_root() / "rules" / "中练度标准.yysrule.json"

# 面板里显示的属性名 -> 规则里的属性标识符。
# 「暴击伤害」在平安志里简写成「暴伤」，两个都收进来。
ATTR_NAME_TO_ID = {
    "暴击伤害": "crit_damage",
    "暴伤": "crit_damage",
    "效果命中": "effect_hit",
    "效果抵抗": "effect_resist",
    "攻击加成": "attack_rate",
    "防御加成": "defense_rate",
    "生命加成": "hp_rate",
    "攻击": "attack_flat",
    "防御": "defense_flat",
    "生命": "hp_flat",
    "速度": "speed",
    "暴击": "crit_rate",
}

ATTR_ID_TO_NAME = {
    "crit_damage": "暴击伤害",
    "effect_hit": "效果命中",
    "effect_resist": "效果抵抗",
    "attack_rate": "攻击加成",
    "defense_rate": "防御加成",
    "hp_rate": "生命加成",
    "attack_flat": "攻击",
    "defense_flat": "防御",
    "hp_flat": "生命",
    "speed": "速度",
    "crit_rate": "暴击",
}

# 号位由主属性唯一确定的那些；剩下三个比例型主属性在 2/4/6 号位都可能出现。
FIXED_SLOT_BY_MAIN = {
    "attack_flat": 1,
    "speed": 2,
    "defense_flat": 3,
    "effect_hit": 4,
    "effect_resist": 4,
    "hp_flat": 5,
    "crit_rate": 6,
    "crit_damage": 6,
}
AMBIGUOUS_SLOTS_BY_MAIN = {
    "attack_rate": (2, 4, 6),
    "defense_rate": (2, 4, 6),
    "hp_rate": (2, 4, 6),
}


def slot_candidates(main_attr: str | None) -> tuple[int, ...]:
    """由主属性推出可能的号位。比例型主属性返回 (2, 4, 6)。"""
    if main_attr is None:
        return (1, 2, 3, 4, 5, 6)
    if main_attr in FIXED_SLOT_BY_MAIN:
        return (FIXED_SLOT_BY_MAIN[main_attr],)
    if main_attr in AMBIGUOUS_SLOTS_BY_MAIN:
        return AMBIGUOUS_SLOTS_BY_MAIN[main_attr]
    return (1, 2, 3, 4, 5, 6)


@dataclass(frozen=True)
class Rule:
    id: str
    name: str
    core: frozenset[str]
    general: frozenset[str]
    core_weight: float
    general_weight: float
    enhancement_core_weight: float
    enhancement_general_weight: float
    keep_score: float
    observe_score: float
    discard_score: float
    quality: frozenset[int]
    slots: frozenset[int]
    slot_main_pairs: tuple[tuple[int, str], ...]
    set_ids: frozenset[str]

    def matches(self, soul: "SoulInfo", slot: int | None, ignore_set: bool = False) -> bool:
        if self.quality and soul.quality not in self.quality:
            return False
        if slot is None or (self.slots and slot not in self.slots):
            return False
        if self.slot_main_pairs:
            if soul.main_attr is None:
                return False
            if (slot, soul.main_attr) not in self.slot_main_pairs:
                return False
        if not ignore_set and self.set_ids and soul.set_id not in self.set_ids:
            return False
        return True


@dataclass(frozen=True)
class AttributeEntry:
    """一条属性。enhancement_count 是被强化了几次（面板右侧徽章上的数字）。"""

    attr_id: str
    enhancement_count: int = 0


@dataclass
class SoulInfo:
    attributes: list[AttributeEntry] = field(default_factory=list)
    set_id: str | None = None
    slot: int | None = None
    quality: int = 6
    main_attr: str | None = None


@dataclass(frozen=True)
class Contribution:
    attr_id: str
    name: str
    kind: str          # core / general
    enhancement_count: int
    weight: float      # 每次强化加的分
    score: float       # 这条属性的总分


@dataclass(frozen=True)
class ScoreResult:
    score: float
    rule_id: str
    rule_name: str
    verdict: str                       # 留 / 观察 / 弃
    contributions: tuple[Contribution, ...] = ()
    slot: int | None = None
    slot_inferred: bool = False
    set_verified: bool = True          # False = 没识别出套装，方案是按最乐观情况挑的
    candidates: tuple[tuple[str, float, str], ...] = ()   # (方案名, 分数, 建议) 备选

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "verdict": self.verdict,
            "slot": self.slot,
            "slot_inferred": self.slot_inferred,
            "set_verified": self.set_verified,
            "candidates": [
                {"rule_name": name, "score": score, "verdict": verdict}
                for name, score, verdict in self.candidates
            ],
            "contributions": [
                {
                    "attribute": c.name,
                    "kind": c.kind,
                    "enhancement_count": c.enhancement_count,
                    "weight": c.weight,
                    "score": c.score,
                }
                for c in self.contributions
            ],
        }


def load_rules(path: str | Path = DEFAULT_RULE_PATH) -> list[Rule]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rules: list[Rule] = []
    for raw in data["rules"]:
        sel = raw["selector"]
        sc = raw["scoring"]
        rules.append(
            Rule(
                id=raw["id"],
                name=raw["name"],
                core=frozenset(sc["coreAttributes"]),
                general=frozenset(sc["generalAttributes"]),
                core_weight=float(sc["coreAttributeWeight"]),
                general_weight=float(sc["generalAttributeWeight"]),
                enhancement_core_weight=float(sc["enhancementCoreWeight"]),
                enhancement_general_weight=float(sc["enhancementGeneralWeight"]),
                keep_score=float(sc["keepScore"]),
                observe_score=float(sc["observeScore"]),
                discard_score=float(sc["discardScore"]),
                quality=frozenset(sel.get("quality") or []),
                slots=frozenset(sel.get("slots") or []),
                slot_main_pairs=tuple(
                    (int(p["slot"]), p["mainAttribute"]) for p in (sel.get("slotMainPairs") or [])
                ),
                set_ids=frozenset(sel.get("setIds") or []),
            )
        )
    return rules


def rule_score(rule: Rule, attributes: list[AttributeEntry]) -> tuple[float, list[Contribution]]:
    """按方案给一组属性打分。逐条属性累加，同名属性会分别计分。"""
    total = 0.0
    contributions: list[Contribution] = []
    for entry in attributes:
        count = entry.enhancement_count or 0
        if entry.attr_id in rule.core:
            weight, kind = rule.enhancement_core_weight, "core"
            base = rule.core_weight
        elif entry.attr_id in rule.general:
            weight, kind = rule.enhancement_general_weight, "general"
            base = rule.general_weight
        else:
            continue
        value = base + count * weight
        total += value
        contributions.append(
            Contribution(
                attr_id=entry.attr_id,
                name=ATTR_ID_TO_NAME.get(entry.attr_id, entry.attr_id),
                kind=kind,
                enhancement_count=count,
                weight=weight,
                score=value,
            )
        )
    return total, contributions


def _verdict(rule: Rule, score: float) -> str:
    if score >= rule.keep_score:
        return "留"
    if score >= rule.observe_score:
        return "观察"
    return "弃"


def score_soul(soul: SoulInfo, rules: list[Rule]) -> ScoreResult | None:
    """给一个御魂打分，返回分数最高的那个方案。没有方案命中时返回 None。

    soul.slot 为空时，用主属性推断候选号位，逐个试、取最高分，并在结果里标记
    slot_inferred——比例型主属性（攻击加成/防御加成/生命加成）在 2/4/6 号位都会
    出现，界面上又看不到号位，只能这么兜。

    soul.set_id 为空（没识别出套装）时，跳过选择器里的套装条件，等于把所有套装
    的方案都拿来试、取最高分——这是个乐观上界，结果里 set_verified 标 False，
    同时给出各备选方案的分数，方便判断这个分数依赖不依赖某个特定套装。
    """
    if soul.slot is not None:
        candidates = (soul.slot,)
        inferred = False
    else:
        candidates = slot_candidates(soul.main_attr)
        # 主属性能唯一确定号位时（速度→2、暴击→6、固定攻击/防御/生命→1/3/5）不算推断
        inferred = len(candidates) > 1

    ignore_set = soul.set_id is None

    best_rule: Rule | None = None
    best_slot: int | None = None
    best_total = 0.0
    best_contributions: list[Contribution] = []
    scored: dict[str, tuple[float, str]] = {}
    for slot in candidates:
        for rule in rules:
            if not rule.matches(soul, slot, ignore_set=ignore_set):
                continue
            total, contributions = rule_score(rule, soul.attributes)
            scored[rule.name] = (total, _verdict(rule, total))
            if best_rule is not None and total <= best_total:
                continue
            best_rule, best_slot, best_total, best_contributions = rule, slot, total, contributions

    if best_rule is None:
        return None

    ranked = sorted(scored.items(), key=lambda kv: kv[1][0], reverse=True)
    return ScoreResult(
        score=best_total,
        rule_id=best_rule.id,
        rule_name=best_rule.name,
        verdict=_verdict(best_rule, best_total),
        contributions=tuple(best_contributions),
        slot=best_slot,
        slot_inferred=inferred,
        set_verified=not ignore_set,
        candidates=tuple((name, score, verdict) for name, (score, verdict) in ranked[:5]),
    )


def entries_from_parsed(
    indexed: list[tuple[int, dict]], badge_counts: dict[int, int] | None = None
) -> list[AttributeEntry]:
    """把 read_panel 解析出的属性行转成评分用的属性条目。

    indexed 每项是 (行下标, parse.AttrLine.to_dict())，行下标是徽章识别用的原始行号；
    badge_counts 以该行下标为键给出强化次数。主属性不参与评分，直接跳过。
    """
    badge_counts = badge_counts or {}
    entries: list[AttributeEntry] = []
    for index, item in indexed:
        if item.get("kind") == "main":
            continue
        attr_id = ATTR_NAME_TO_ID.get(item.get("name", ""))
        if attr_id is None:
            continue
        entries.append(AttributeEntry(attr_id=attr_id, enhancement_count=int(badge_counts.get(index, 0))))
    return entries


def main_attribute(indexed: list[tuple[int, dict]]) -> str | None:
    """取主属性的属性标识符，用来推断号位。"""
    for _, item in indexed:
        if item.get("kind") == "main":
            return ATTR_NAME_TO_ID.get(item.get("name", ""))
    return None
