"""从一帧画面跑到「御魂报告」的完整识别链路。

命令行 tools/read_panel.py 和图形界面 tools/gui.py 都调这里，两边只是把同一份
报告按不同方式呈现（一个打印表格、一个填进窗口），识别流程只有这一份。

支持两种界面：

  * 御魂详情页——仓库 / 列表右侧弹出的详情面板，只显示选中的那一件。先判这一种
    （src/detail.py）：面板定位靠「一块足够大的浅蓝区域」，批量页没有这样一块，
    所以不会误判。
  * 御魂批量强化的结果页。一屏 N 张卡片就用网格切（find_cards），只剩一张时同样
    按卡片走，这样每张卡片右侧的强化次数徽章都能读到。网格检测不到时才退回按面板
    比例切行（老路径，没有徽章，套装改为在左上角区域直接搜图标）。

套装：详情页直接读面板上的套装名文字；批量页靠卡片头部左侧的圆形图标识别
（src/seticon.py）。识别不出时 set_id 传 None，评分取「所有套装方案里最高的
那个」，结果里 set_verified=False。

号位：详情页直接读「壹号位」文本；批量页优先取图标背后的角标
（src/seticon.slot_from_icon），角标判不出或与主属性矛盾时退回「主属性推断」，
结果里标 slot_inferred。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.badge import read_badges
from src.detail import analyze_detail
from src.ocr import OcrLine, read_region, read_rows
from src.panel import Card, Rect, find_cards, find_panel_card, find_rows, panel_roi
from src.parse import parse_lines_indexed
from src.score import (
    ATTR_ID_TO_NAME,
    Rule,
    ScoreResult,
    SoulInfo,
    entries_from_parsed,
    main_attribute,
    score_soul,
    slot_candidates,
)
from src.seticon import SetMatch, analyze, search, slot_from_icon

# 6 星御魂才进评分方案（28 条方案的 quality 都是 [6]）
QUALITY = 6

# 界面形态：批量强化网格 / 单个御魂面板（能读徽章）/ 单个御魂面板（读不到徽章）/
# 御魂详情面板
MODE_BATCH = "batch"
MODE_SINGLE_CARD = "single_card"
MODE_SINGLE_ROWS = "single_rows"
MODE_SOUL_DETAIL = "soul_detail"

# 用来区分「调用方没传」和「调用方传了 None（就是没识别出来）」
_UNSET = object()


class AnalysisError(RuntimeError):
    """画面里找不到可识别的内容。"""


@dataclass
class SoulReport:
    """一件御魂的全部识别结果。"""

    index: int                          # 读序（0 起），就是界面上的序号 - 1
    lines: list[OcrLine]                # 这个区域 OCR 出来的行（含没解析成功的）
    kinds: list[str]                    # 每行的粗分类 main / sub / innate
    indexed: list[tuple[int, dict]]     # (行下标, parse.AttrLine.to_dict())
    badges: dict[int, int]              # 行下标 -> 这条属性被强化了几次
    card: Card | None                   # 卡片网格给出的卡片；没有网格时为 None
    set_match: SetMatch | None
    main_attr: str | None               # 主属性的属性标识符
    slot: int | None
    result: ScoreResult | None
    warnings: list[str] = field(default_factory=list)

    @property
    def set_name(self) -> str | None:
        return self.set_match.name if self.set_match else None

    @property
    def box(self) -> Rect | None:
        """这件御魂在画面上的外接方框，用来在预览图上画框。"""
        if self.card is not None:
            return self.card.rect
        if not self.lines:
            return None
        x0 = min(line.rect.x for line in self.lines)
        y0 = min(line.rect.y for line in self.lines)
        x1 = max(line.rect.x1 for line in self.lines)
        y1 = max(line.rect.y1 for line in self.lines)
        return Rect(x=x0, y=y0, w=x1 - x0, h=y1 - y0)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "set": self.set_name,
            "set_score": round(self.set_match.score, 3) if self.set_match else None,
            "main_attribute": ATTR_ID_TO_NAME.get(self.main_attr or ""),
            "badges": {str(key): value for key, value in sorted(self.badges.items())},
            "attributes": [{"row": row, **item} for row, item in self.indexed],
            "score": self.result.to_dict() if self.result else None,
        }


@dataclass
class PanelAnalysis:
    """一屏画面的识别结果。"""

    mode: str
    souls: list[SoulReport]
    size: tuple[int, int]               # 客户区 / 图片的 (宽, 高)

    def to_dict(self, window: str, captured_at: str) -> dict:
        return {
            "window": window,
            "captured_at": captured_at,
            "mode": self.mode,
            "souls": [soul.to_dict() for soul in self.souls],
        }


def resolve_slot(
    frame: np.ndarray, icon: Rect | None, main_attr: str | None
) -> tuple[int | None, str | None]:
    """号位优先取图标角标，返回 (号位, 冲突提示)。

    判不出号位时返回 (None, None)，上层退回主属性推断。比例型主属性（攻击/防御/
    生命加成）在 2/4/6 号位都会出现，「矛盾」只可能发生在固定主属性上——那说明
    角标判错了，宁可不信它，返回 (None, 提示)。
    """
    if icon is None:
        return None, None
    slot = slot_from_icon(frame, icon)
    if slot is None:
        return None, None
    allowed = slot_candidates(main_attr)
    if slot not in allowed:
        main_name = ATTR_ID_TO_NAME.get(main_attr or "", main_attr or "未知")
        return None, (
            f"图标角标判为 {slot} 号位，但主属性 {main_name} 只可能出现在 "
            f"{'/'.join(str(item) for item in allowed)} 号位，改按主属性推断"
        )
    return slot, None


def _build_soul(
    frame: np.ndarray,
    index: int,
    lines: list[OcrLine],
    kinds: list[str],
    card: Card | None,
    rules: list[Rule],
    *,
    set_match: SetMatch | None | object = _UNSET,
    slot: int | None | object = _UNSET,
    badges: dict[int, int] | object = _UNSET,
) -> SoulReport:
    """把一组已经切好的行拼成报告。

    批量卡片走默认路径：套装按头部图标识别（没卡片信息时退化成在左上角搜），
    号位按图标角标判、判不出再按主属性推断，徽章按卡片右侧的亮数字簇读。详情面板
    则把 set_match / slot / badges 直接传进来——那边全是读面板上的东西，比看图标
    可靠（见 src/detail.py）。
    """
    indexed = [(i, parsed.to_dict()) for i, parsed in parse_lines_indexed(lines, kinds=kinds)]
    if badges is _UNSET:
        badges = read_badges(frame, card) if card is not None else {}

    # 卡片上直接按头部图标识别；没卡片信息时（只剩一张卡）退化成在左上角区域搜
    icon: Rect | None = None
    if set_match is _UNSET:
        if card is not None:
            set_match, icon = analyze(frame, card)
        else:
            set_match = search(frame)

    main_attr = main_attribute(indexed)
    warning = None
    if slot is _UNSET:
        slot, warning = resolve_slot(frame, icon, main_attr)
    info = SoulInfo(
        attributes=entries_from_parsed(indexed, badges),
        set_id=set_match.name if set_match else None,
        slot=slot,
        quality=QUALITY,
        main_attr=main_attr,
    )
    return SoulReport(
        index=index,
        lines=lines,
        kinds=kinds,
        indexed=indexed,
        badges=badges,
        card=card,
        set_match=set_match,
        main_attr=main_attr,
        slot=slot,
        result=score_soul(info, rules),
        warnings=[warning] if warning else [],
    )


def analyze_frame(
    frame: np.ndarray,
    rules: list[Rule],
    threshold: int = 100,
    scale: int = 2,
) -> PanelAnalysis:
    """跑完整条链路。画面里什么都没有时抛 AnalysisError。"""
    height, width = frame.shape[:2]

    # 详情页先判：面板定位很快（只是一次颜色连通域），只有真找到面板才会去 OCR，
    # 所以批量页不会为这一步多花时间。找不到面板就直接走下面的批量流程。
    detail = analyze_detail(frame, scale=scale)
    if detail is not None:
        report = _build_soul(
            frame,
            0,
            detail.lines,
            detail.kinds,
            detail.card,
            rules,
            set_match=detail.set_match,
            slot=detail.slot,
            badges=detail.badges,
        )
        return PanelAnalysis(mode=MODE_SOUL_DETAIL, souls=[report], size=(width, height))

    cards = find_cards(frame, threshold=threshold)

    # units: (序号, OCR 行, 行类型, 卡片——没有卡片网格时为 None)
    units: list[tuple[int, list[OcrLine], list[str], Card | None]] = []
    if cards:
        mode = MODE_BATCH
        for index, card in enumerate(cards):
            lines = read_region(frame, card.rect, list(card.rows), scale=scale)
            units.append((index, lines, [row.kind for row in card.rows], card))
    else:
        roi = panel_roi(frame)
        # 网格找不到时，先按面板行投影反推卡片框——剩余的那张卡就摆在左上角第一格，
        # 反推出来同样能读徽章、能按头部图标识别套装。
        card = find_panel_card(frame, roi=roi, threshold=threshold)
        if card is not None:
            lines = read_region(frame, card.rect, list(card.rows), scale=scale)
            units.append((0, lines, [row.kind for row in card.rows], card))
            mode = MODE_SINGLE_CARD
        else:
            rows = find_rows(frame, roi=roi, threshold=threshold)
            if not rows:
                raise AnalysisError("没检测到任何文字行，确认游戏是否停在强化结果界面。")
            lines = read_rows(frame, rows, scale=scale)
            units.append((0, lines, [row.kind for row in rows], None))
            mode = MODE_SINGLE_ROWS

    souls = [
        _build_soul(frame, index, lines, kinds, card, rules)
        for index, lines, kinds, card in units
    ]
    return PanelAnalysis(mode=mode, souls=souls, size=(width, height))
