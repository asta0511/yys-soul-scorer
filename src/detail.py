"""御魂详情页：识别仓库 / 御魂列表右侧弹出的详情面板里选中的那一件御魂。

和批量强化结果页不是一回事。详情面板的底色是浅蓝（灰度约 192）、文字是白或深蓝，
对比度极低，按亮度做行投影切不出行，所以反过来做：先把整个面板 OCR 一次，再按
文本框的中心 y 把结果归成行。

面板本身靠「一块足够大的浅蓝连通区域」定位（落在右半屏、宽约客户区的四分之一、
高约三分之二）。批量强化结果页没有这样一块，所以这个判据能把两种界面干净地分开。

号位和套装名都直接读面板上的文字：「壹号位」映射到 1，套装名跟 data/soul_icons
的文件名（也就是规则里的 setIds）比对。这比批量页靠图标角标 / 图标模板匹配可靠
得多——详情面板的字是批量卡片的三倍大，实测 OCR 置信度都在 0.99 以上。套装名读
不出来时留空，评分退回「所有套装方案里最高的那个」。

标定基准：阴阳师 PC 端，客户区 1198x675（面板 bbox 实测 552,147 293x453）。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.badge import read_digit_in_box
from src.ocr import OcrLine, read_boxes
from src.panel import BASE_HEIGHT, Card, Rect, TextRow
from src.parse import match_name
from src.seticon import SetMatch, set_names

# ---- 面板定位 ----

# 面板底色：灰度落在这一带，且蓝通道明显高于红通道（浅蓝）
PANEL_GRAY = (165, 205)
PANEL_BLUE_OVER_RED = 20

# 面板的形状与位置约束，用来把左上的状态栏、右侧的按钮列挡在外面
PANEL_MIN_AREA_RATIO = 0.05         # 占客户区面积的比例下限
PANEL_WIDTH_RATIO = (0.15, 0.40)    # 宽度占客户区宽度的比例区间
PANEL_HEIGHT_RATIO = (0.40, 0.95)   # 高度占客户区高度的比例区间
PANEL_MIN_X_RATIO = 0.30            # 面板左边界必须在客户区这个位置以右

# ---- 面板内 ----

# 归行容差：文本框中心 y 相差不到「行高的这个比例」就算同一行
ROW_TOL_RATIO = 0.6

# ---- 属性行右侧的强化次数徽章 ----
#
# 和批量卡片上的徽章正好相反：这边是深色圆盘（灰度约 105）配白字，面板底色却是
# 浅蓝（约 192），所以按颜色找的是「暗块」而不是「亮点」。
#
# 另一个坑是裁剪范围：裁得松（数字质心 ± 11x13）时 OCR 会把「圆盘 + 数字」整体
# 读成圈码（①），4 个徽章里错 2 个；贴着圆盘裁（16x16）则四个全对。所以先找圆盘，
# 再拿圆盘的框去识别。
BADGE_DISC_GRAY = 165
BADGE_DISC_DIAM = (10, 30)      # 圆盘外接框长边（像素，基准分辨率 790 高）
BADGE_DISC_FILL = 0.5           # 短边 / 长边 的下限，滤掉细长的噪块
BADGE_DISC_PAD = 1              # 裁的时候往外放几个像素
BADGE_SEARCH_X = (1, 24)        # 相对卡片右边界向右的搜索范围
BADGE_SEARCH_PAD_Y = 5          # 搜索区在副属性行上下各外扩多少
BADGE_ASSIGN_TOL_RATIO = 0.6    # 与行中心的垂直距离超过这个比例 × 行高就不归给它

# 号位文本「壹号位」…「陆号位」
SLOT_LABELS = {"壹": 1, "贰": 2, "叁": 3, "肆": 4, "伍": 5, "陆": 6}
SLOT_SUFFIX = "号位"

# 面板下半部分的套装说明与按钮，不能当属性行解析（否则会刷一堆 unparsed 警告）
DESC_MARKERS = ("件套", "概率")
BUTTON_TEXT = "分享"


@dataclass(frozen=True)
class DetailPanel:
    """详情面板的识别结果。"""

    panel: Rect
    card: Card                      # 属性行的框，右边界压在属性值右侧供读徽章
    lines: list[OcrLine]            # 主属性 + 副属性行（已排除说明文字与按钮）
    kinds: list[str]
    badges: dict[int, int]          # 行下标 -> 这条属性强化了几次
    slot: int | None
    set_match: SetMatch | None


def find_panel(frame: np.ndarray) -> Rect | None:
    """定位右半屏那块浅蓝详情面板；没有返回 None。

    先按颜色抠出浅蓝掩码，再取连通域里面积最大且形状合理的那一块。面板四周有一圈
    描边会把区域切断，所以先做一次闭运算黏回去。
    """
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blue, green, red = (frame[:, :, k].astype(np.int32) for k in range(3))
    low, high = PANEL_GRAY
    mask = (
        (gray >= low)
        & (gray <= high)
        & (blue - red > PANEL_BLUE_OVER_RED)
        & (blue > green)
    ).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best: Rect | None = None
    for label in range(1, count):
        x, y, w, h, area = (int(stats[label, k]) for k in range(5))
        if area < PANEL_MIN_AREA_RATIO * width * height:
            continue
        if not PANEL_WIDTH_RATIO[0] <= w / width <= PANEL_WIDTH_RATIO[1]:
            continue
        if not PANEL_HEIGHT_RATIO[0] <= h / height <= PANEL_HEIGHT_RATIO[1]:
            continue
        if x / width < PANEL_MIN_X_RATIO:
            continue
        if best is None or area > best.w * best.h:
            best = Rect(x=x, y=y, w=w, h=h)
    return best


def _group_rows(lines: list[OcrLine], tol: float) -> list[list[OcrLine]]:
    """按中心 y 把文本框归成行，行内按 x 从左到右排。"""
    rows: list[list[list]] = []
    for line in sorted(lines, key=lambda item: (item.rect.y + item.rect.y1) / 2):
        center = (line.rect.y + line.rect.y1) / 2
        if rows and center - rows[-1][0] <= tol:
            rows[-1][1].append(line)
        else:
            rows.append([center, [line]])
    return [sorted(items, key=lambda item: item.rect.x) for _, items in rows]


def _row_kind(text: str, names: tuple[str, ...]) -> str:
    """粗分一行是号位 / 套装 / 属性 / 说明 / 按钮。

    顺序有讲究：说明文字里也会出现属性名（「2件套属性：防御加成30%」），必须先
    按「件套/概率」把它们挡掉，再谈匹配属性名。
    """
    if SLOT_SUFFIX in text and any(label in text for label in SLOT_LABELS):
        return "slot"
    if BUTTON_TEXT in text:
        return "button"
    if any(marker in text for marker in DESC_MARKERS):
        return "desc"
    if match_name(text) is not None:
        return "attr"
    if any(name in text for name in names):
        return "set"
    return "other"


def _slot_from_text(text: str) -> int | None:
    for label, slot in SLOT_LABELS.items():
        if label in text:
            return slot
    return None


def _set_from_text(text: str, names: tuple[str, ...]) -> str | None:
    """取文本里出现的最长套装名。"""
    hits = [name for name in names if name in text]
    return max(hits, key=len) if hits else None


def read_badges(frame: np.ndarray, card: Card) -> dict[int, int]:
    """读属性行右侧的强化次数徽章，返回 {行下标: 次数}。

    只在副属性行的 y 范围内找：主属性的字更大、伸得更右（实测到 x830，副属性都止
    于 810），会被当成一个深色块。

    找的是深色圆盘（见 BADGE_DISC_GRAY），再拿圆盘的外接框去 OCR —— 裁剪范围一
    放大，OCR 就会把整枚徽章读成圈码。
    """
    sub = [(index, row) for index, row in enumerate(card.rows) if row.kind == "sub"]
    if not sub:
        return {}

    scale = frame.shape[0] / BASE_HEIGHT
    x0 = card.rect.x1 + round(BADGE_SEARCH_X[0] * scale)
    x1 = card.rect.x1 + round(BADGE_SEARCH_X[1] * scale)
    pad = round(BADGE_SEARCH_PAD_Y * scale)
    y0 = max(min(row.rect.y for _, row in sub) - pad, 0)
    y1 = min(max(row.rect.y1 for _, row in sub) + pad, frame.shape[0])
    if x1 > frame.shape[1] or x1 <= x0 or y1 <= y0:
        return {}

    gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    mask = (gray <= BADGE_DISC_GRAY).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    low, high = (round(value * scale) for value in BADGE_DISC_DIAM)
    tolerance = float(np.median([row.rect.h for _, row in sub]))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    result: dict[int, int] = {}
    for label in range(1, count):
        bx, by, bw, bh = (int(stats[label, k]) for k in range(4))
        if not low <= max(bw, bh) <= high:
            continue
        if min(bw, bh) < BADGE_DISC_FILL * max(bw, bh):
            continue

        value = read_digit_in_box(
            frame,
            Rect(
                x=x0 + bx - BADGE_DISC_PAD,
                y=y0 + by - BADGE_DISC_PAD,
                w=bw + 2 * BADGE_DISC_PAD,
                h=bh + 2 * BADGE_DISC_PAD,
            ),
        )
        if value is None:
            continue

        center = y0 + by + bh / 2
        index, distance = min(
            (
                (i, abs(center - (row.rect.y + row.rect.y1) / 2))
                for i, row in sub
            ),
            key=lambda pair: pair[1],
        )
        # 离得太远说明这个暗块不是任何一行的徽章（比如面板上的装饰）
        if distance <= BADGE_ASSIGN_TOL_RATIO * tolerance:
            result[index] = value
    return result


def _text_row(items: list[OcrLine], kind: str) -> TextRow:
    x0 = min(item.rect.x for item in items)
    y0 = min(item.rect.y for item in items)
    x1 = max(item.rect.x1 for item in items)
    y1 = max(item.rect.y1 for item in items)
    return TextRow(rect=Rect(x=x0, y=y0, w=x1 - x0, h=y1 - y0), kind=kind)


def analyze_detail(frame: np.ndarray, scale: int = 2) -> DetailPanel | None:
    """识别详情面板里选中的御魂；这块画面不是详情面板时返回 None。"""
    panel = find_panel(frame)
    if panel is None:
        return None

    boxes = read_boxes(frame, panel, scale=scale)
    if not boxes:
        return None

    heights = sorted(item.rect.h for item in boxes)
    tol = ROW_TOL_RATIO * heights[len(heights) // 2]

    names = set_names()
    slot: int | None = None
    set_match: SetMatch | None = None
    attr_rows: list[list[OcrLine]] = []
    for items in _group_rows(boxes, tol):
        text = "".join(item.text for item in items)
        kind = _row_kind(text, names)
        if kind == "slot":
            slot = _slot_from_text(text)
        elif kind == "set":
            name = _set_from_text(text, names)
            if name is not None:
                set_match = SetMatch(name=name, score=1.0, margin=1.0, source="text")
        elif kind == "attr":
            attr_rows.append(items)

    if not attr_rows:
        return None

    rows = [
        _text_row(items, "main" if index == 0 else "sub")
        for index, items in enumerate(attr_rows)
    ]
    lines = [
        OcrLine(
            rect=row.rect,
            text="".join(item.text for item in items),
            score=min(item.score for item in items),
        )
        for row, items in zip(rows, attr_rows)
    ]

    # 徽章紧跟在属性值右边，所以卡片右边界要压在「副属性的值右边界」上——主属性
    # 字号更大、伸得更右（实测主属性到 830、副属性都止于 810），拿它当边界的话
    # 搜索窗就滑过了徽章。
    right_edges = [
        max(item.rect.x1 for item in items)
        for row, items in zip(rows, attr_rows)
        if row.kind == "sub"
    ]
    if not right_edges:
        right_edges = [max(item.rect.x1 for item in items) for items in attr_rows]
    left = min(row.rect.x for row in rows)
    right = max(right_edges)
    top = rows[0].rect.y
    bottom = rows[-1].rect.y1

    card = Card(
        rect=Rect(x=left, y=top, w=right - left, h=bottom - top),
        rows=tuple(rows),
        header=Rect(x=left, y=top, w=right - left, h=0),
    )
    return DetailPanel(
        panel=panel,
        card=card,
        lines=lines,
        kinds=[row.kind for row in rows],
        badges=read_badges(frame, card),
        slot=slot,
        set_match=set_match,
    )
