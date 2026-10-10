"""把 OCR 出来的一行文本拆成「属性名 / 旧值 / 最终值」。

为什么不靠分隔符切：面板里那个蓝色双箭头 «» 很小，OCR 结果不稳定——实测同一屏里
它会被读成 ">"、"》"，或者直接丢掉（"防御加成2.75%11.02%"）。所以改成按内容解析：

    1. 去掉空格，先把「固有属性：」前缀摘掉并标记为 innate；
    2. 用已知属性名表做最长前缀匹配，拿到属性名（能区分「防御」和「防御加成」）；
    3. 剩下的部分里抓所有数字，第一个是旧值、最后一个是最终值；只有一个数字时
       说明这条属性本次没被强化，旧值 = 最终值。

数值解析：OCR 偶尔把小数点读成全角或逗号，统一归一化后再取数。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from src.ocr import OcrLine

# 御魂属性名全表（11 种）。匹配时按长度降序，保证「暴击伤害」不会被「暴击」截胡。
ATTR_NAMES = (
    "暴击伤害",
    "效果命中",
    "效果抵抗",
    "攻击加成",
    "防御加成",
    "生命加成",
    "攻击",
    "防御",
    "生命",
    "速度",
    "暴击",
)
_NAMES_BY_LEN = tuple(sorted(ATTR_NAMES, key=len, reverse=True))

INNATE_PREFIX = "固有属性"

# 批量强化里，本次被强化过的副属性左侧会挂一个绿色「新」标签，OCR 会把它读进文本
NEW_TAG = "新"

# 游戏内所有属性值都是两位小数（防御 14.00、暴击 3.80%）。据此切分数值，而不是
# 靠中间的 «» 箭头——那个箭头 OCR 经常漏掉，漏掉后两个数会直接相邻（"生命98.99
# 192.41" 去掉空格变成 "98.99192.41"），用通用的「整数+可选小数」正则会把它们
# 并成一个数（98.99192）。限定小数位数 1~2 位即可正确切开。纯整数（无小数点）
# 的情况由第二分支兜底。
_NUM_RE = re.compile(r"(\d+\.\d{1,2}|\d+)\s*(%?)")
_FULLWIDTH = str.maketrans({"％": "%", "．": ".", "，": ",", "０": "0", "１": "1",
                            "２": "2", "３": "3", "４": "4", "５": "5", "６": "6",
                            "７": "7", "８": "8", "９": "9"})


@dataclass(frozen=True)
class AttrLine:
    kind: str        # main 主属性 / sub 副属性 / innate 固有属性
    name: str        # 属性名，如「防御加成」
    old: float       # 旧值（强化前）
    final: float     # 最终值（强化后）
    unit: str        # '%' 或 ''
    score: float     # OCR 置信度
    raw: str         # OCR 原文，留作排查

    @property
    def upgraded(self) -> bool:
        return self.old != self.final

    def to_dict(self) -> dict:
        data = asdict(self)
        data["upgraded"] = self.upgraded
        return data


def _normalize(text: str) -> str:
    return text.translate(_FULLWIDTH).replace(" ", "").replace("\u3000", "")


def match_name(text: str) -> str | None:
    for name in _NAMES_BY_LEN:
        if text.startswith(name):
            return name
    return None


def extract_values(text: str) -> list[tuple[float, str]]:
    values: list[tuple[float, str]] = []
    for number, unit in _NUM_RE.findall(text):
        try:
            values.append((float(number), unit))
        except ValueError:
            continue
    return values


def parse_line(line: OcrLine, kind: str) -> AttrLine | None:
    text = _normalize(line.text)
    if not text:
        return None

    if text.startswith(INNATE_PREFIX):
        kind = "innate"
        text = text[len(INNATE_PREFIX):].lstrip("：:")

    # 「新」标签可能被读成多个字（"新新"），只要剥掉后能匹配上属性名就一直剥
    while text.startswith(NEW_TAG) and match_name(text[len(NEW_TAG):]) is not None:
        text = text[len(NEW_TAG):]

    name = match_name(text)
    if name is None:
        return None

    values = extract_values(text[len(name):])
    if not values:
        return None

    old, old_unit = values[0]
    final, final_unit = values[-1]
    unit = final_unit or old_unit
    return AttrLine(
        kind=kind,
        name=name,
        old=old,
        final=final,
        unit=unit,
        score=line.score,
        raw=line.text,
    )


def parse_lines_indexed(
    lines: list[OcrLine], kinds: list[str] | None = None
) -> list[tuple[int, AttrLine]]:
    """跟 parse_lines 一样，但保留每条属性在 lines 里的原始下标。

    徽章的强化次数是按「卡片行号」读出来的，而 parse_lines 会丢掉解析失败的行、
    下标就错位了，所以评分链路走这个函数。
    """
    result: list[tuple[int, AttrLine]] = []
    for index, line in enumerate(lines):
        kind = kinds[index] if kinds else ("main" if index == 0 else "sub")
        parsed = parse_line(line, kind)
        if parsed is not None:
            result.append((index, parsed))
    return result


def parse_lines(lines: list[OcrLine], kinds: list[str] | None = None) -> list[AttrLine]:
    """kinds 由 panel.find_rows 给出的粗分类（首行 main、其余 sub）。

    解析不出来的行会被丢弃，其原文通过 unparsed() 取回，方便排查是 OCR 读错还是
    出现了属性名表里没有的新属性。
    """
    return [parsed for _, parsed in parse_lines_indexed(lines, kinds=kinds)]


def unparsed(lines: list[OcrLine], kinds: list[str] | None = None) -> list[OcrLine]:
    """返回没能解析成属性的行，供上层提示用户。"""
    result: list[OcrLine] = []
    for index, line in enumerate(lines):
        kind = kinds[index] if kinds else ("main" if index == 0 else "sub")
        # 空文本行只是没读到字，不值得当「解析失败」报警
        if not line.text.strip():
            continue
        if parse_line(line, kind) is None:
            result.append(line)
    return result


def format_value(value: float, unit: str) -> str:
    """统一两位小数，跟游戏内显示保持一致（防御 104.00、暴击 11.02%）。"""
    return f"{value:.2f}{unit}"
