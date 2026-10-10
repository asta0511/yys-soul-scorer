"""御魂强化「详细属性」面板的定位与行切分。

面板在客户区中的位置相对固定（面板左上角锚定，属性条数变化时只向下变高），
所以用比例 ROI 粗定位；再在 ROI 内做「行投影」自动切出每一行文字——行数变化
也能自适应，不用为每一行写死坐标。

行投影原理：面板文字是亮色、面板底色是暗的半透明蓝，逐行统计亮像素数量，
连续超阈值的行就构成一行文字。

横向裁剪范围由实机标定得出，目的是切掉两个会干扰 OCR 的东西：
  - 左侧的面板边框竖线（x≈224），否则会被识别成 "1" 或 "|"
  - 右侧的强化次数徽章圆点（x≥503）与分隔线（x≈502）

标定基准：阴阳师 PC 端，客户区 1402x790。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# 面板搜索区相对客户区的比例 (x0, y0, x1, y1)
# x 已收进边框内侧，y 故意开大以容纳属性条数变化
PANEL_ROI = (0.1626, 0.070, 0.3573, 0.800)

# 标定基准：阴阳师 PC 客户区 1402x790。比例型参数（ROI、峰值占比）天然随分辨率
# 自适应；像素型阈值（最小行高、合并间隙、内缩）都按这个高度标定，换分辨率时用
# _ui_scale() 换算，否则窗口一缩小行高阈值就再也达不到。
BASE_HEIGHT = 790

# 图标行的行高远大于文字行，用它把「御魂图标 + Lv.0 » Lv.15」整块剔除
ICON_BAND_RATIO = 2.0

# ---- 批量强化：卡片网格 ----

# 网格搜索区相对客户区 (x0, y0, x1, y1)，去掉左侧滚动条和右侧「弃置/恢复」按钮列
GRID_ROI = (0.155, 0.05, 0.86, 1.0)

# 采样卡片底色的窗口（必须落在第一张卡片正文内），相对客户区
CARD_BG_SAMPLE = (0.20, 0.28, 0.32, 0.42)

# 卡片底色匹配容差：BGR 三通道绝对差之和
CARD_BG_TOL = 36

# 判定「这一列是卡片」要求卡片底色像素占「卡片正文行数」的比例。
# 两个坑：
#   1) 分母必须是正文行数而不是网格高度——一屏只有一行卡时正文只占网格高度四分之一
#      上下，按网格高度算的话三列里有两列都会被刷掉（实测 1200x676 三张卡只认出一张）。
#   2) 比例不能太高——被强化的副属性左侧挂的绿色「新」标签不是底色，会把卡片最左边
#      那几十像素的计数压低，阈值一高整列左边界就右移，行首字被切掉（实测 0.65 时
#      第一列左边界从 198 右移到 241，「攻击加成」读成「加成」）。
# 0.45~0.55 实测两张图都对，取中点留余量。
CARD_COL_FILL = 0.50

# 找出「卡片正文所在的那几行」：该行命中卡片底色的像素数要够到网格里最高那行的这个比例。
# 卡片下方那片背景的底色和卡片很接近（实测能到网格宽度的六成），必须用高比例把
# 它挡掉，否则它会混进正文行、把没卡的列也投高，相邻两列还会并成一列。
CARD_ROW_FILL = 0.7
# 连一行都到不了「网格宽度的这个比例」，就认为这一屏没有卡片
CARD_ROW_MIN_FILL = 0.15

# 卡片最小宽度（像素），用来滤掉边框、徽章之类的小块
CARD_MIN_WIDTH = 80

# 卡片正文左右各内缩多少像素：左侧避开「新」标签，右侧避开强化次数徽章
CARD_INSET_X = (6, 5)

# 判定一段是「文字行」所需的最小亮度峰值（占卡片宽度比例）。
# 卡片底部的分隔线/圆角边框是一条横贯整列的暗亮曲线，逐行亮度峰值只有 9% 左右；
# 真正的文字行峰值都在 16% 以上。用比例而不是绝对像素，换分辨率时不用重标定。
CARD_ROW_PEAK_RATIO = 0.12


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def x1(self) -> int:
        return self.x + self.w

    @property
    def y1(self) -> int:
        return self.y + self.h

    def to_slice(self) -> tuple[slice, slice]:
        return slice(self.y, self.y1), slice(self.x, self.x1)

    def __str__(self) -> str:
        return f"({self.x},{self.y} {self.w}x{self.h})"


@dataclass(frozen=True)
class TextRow:
    rect: Rect
    kind: str  # 'main' 主属性 / 'sub' 副属性 / 'innate' 固有属性
    text: str = ""


def _ui_scale(frame: np.ndarray) -> float:
    """客户区相对标定基准的缩放系数，用于把像素阈值换算到当前分辨率。"""
    return frame.shape[0] / BASE_HEIGHT


def panel_roi(frame: np.ndarray) -> Rect:
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = PANEL_ROI
    return Rect(
        x=int(width * x0),
        y=int(height * y0),
        w=int(width * (x1 - x0)),
        h=int(height * (y1 - y0)),
    )


def _bright_mask(roi: np.ndarray, threshold: int) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return (gray >= threshold).astype(np.uint8)


def _bands(
    counts: np.ndarray, min_len: int, min_pixels: int, merge_gap: int
) -> list[list[int]]:
    """在一维计数数组上找连续超阈值的段，再把挨得近的段合并。

    counts 是某个方向上的投影计数（行投影或列投影都走这里）。

    顺序很关键：先把所有短段也收进来、按 merge_gap 合并，最后才按 min_len 过滤。
    反过来做的话，一个被装饰元素切出 1~2 像素空隙的长段会被拆成两个短段、双双被丢掉。
    """
    active = counts >= min_pixels

    raw: list[list[int]] = []
    start: int | None = None
    for index, flag in enumerate(active):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            raw.append([start, index])
            start = None
    if start is not None:
        raw.append([start, len(active)])

    merged: list[list[int]] = []
    for band in raw:
        if merged and band[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = band[1]
        else:
            merged.append(band)
    return [band for band in merged if band[1] - band[0] >= min_len]


def _row_bands(
    mask: np.ndarray, min_height: int, min_pixels: int, merge_gap: int
) -> list[list[int]]:
    return _bands(mask.sum(axis=1), min_height, min_pixels, merge_gap)


def _drop_icon_band(bands: list[list[int]]) -> list[list[int]]:
    """首个行段若是图标区（高度远超其余行），剔除它。"""
    if len(bands) < 2:
        return bands
    rest = [b[1] - b[0] for b in bands[1:]]
    median_rest = float(np.median(rest))
    first_height = bands[0][1] - bands[0][0]
    if median_rest > 0 and first_height > ICON_BAND_RATIO * median_rest:
        return bands[1:]
    return bands


def find_rows(
    frame: np.ndarray,
    roi: Rect | None = None,
    threshold: int = 100,
    min_pixels: int = 2,
    min_height: int = 8,
    merge_gap: int = 3,
    pad_y: int = 3,
    drop_icon: bool = True,
) -> list[TextRow]:
    """返回面板内每一行文字的裁剪框，坐标已换算回整帧。

    kind 先按位置粗分：首行 'main'，其余 'sub'。真正的固有属性行要等 OCR 出
    文本后才能确认，用 classify() 再修正。
    """
    roi = roi or panel_roi(frame)
    scale = _ui_scale(frame)
    min_height = max(3, round(min_height * scale))
    min_pixels = max(1, round(min_pixels * scale))
    merge_gap = max(1, round(merge_gap * scale))
    pad_y = max(1, round(pad_y * scale))

    mask = _bright_mask(frame[roi.to_slice()], threshold)

    bands = _row_bands(mask, min_height, min_pixels, merge_gap)
    if drop_icon:
        bands = _drop_icon_band(bands)
    if not bands:
        return []

    rows: list[TextRow] = []
    for index, (start, end) in enumerate(bands):
        y = max(roi.y + start - pad_y, 0)
        y1 = min(roi.y + end + pad_y, frame.shape[0])
        rect = Rect(x=roi.x, y=y, w=roi.w, h=y1 - y)
        rows.append(TextRow(rect=rect, kind="main" if index == 0 else "sub"))

    return rows


def classify(rows: list[TextRow], innate_prefixes: tuple[str, ...] = ("固有属性",)) -> list[TextRow]:
    """OCR 之后按文本内容修正行类型。"""
    result: list[TextRow] = []
    for index, row in enumerate(rows):
        text = row.text.strip()
        if text.startswith(innate_prefixes):
            kind = "innate"
        elif index == 0:
            kind = "main"
        else:
            kind = "sub"
        result.append(TextRow(rect=row.rect, kind=kind, text=row.text))
    return result


def crop(frame: np.ndarray, rect: Rect, scale: int = 1) -> np.ndarray:
    img = frame[rect.to_slice()]
    if scale != 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return img


# ---------------------------------------------------------------- 批量强化网格


@dataclass(frozen=True)
class Card:
    """批量强化界面里的一张御魂卡片。"""

    rect: Rect                       # 卡片正文区域（已去掉边框、图标行、徽章）
    rows: tuple[TextRow, ...]        # 主属性 + 副属性 + 固有属性
    header: Rect                     # 顶部整行「套装图标 + Lv.0 » Lv.15」
    column: int = 0                  # 第几列（0 起）
    row: int = 0                     # 第几行（0 起）


def grid_roi(frame: np.ndarray) -> Rect:
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = GRID_ROI
    return Rect(
        x=int(width * x0),
        y=int(height * y0),
        w=int(width * (x1 - x0)),
        h=int(height * (y1 - y0)),
    )


def _card_bg_color(frame: np.ndarray) -> np.ndarray:
    """估计卡片底色：在第一张卡片正文内取众数颜色。

    取众数而不是均值，是因为窗口里混着文字和图标，均值会被拉偏。
    先把颜色量化到 8 级再统计，避免渐变压成无数个"不同"颜色。
    """
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = CARD_BG_SAMPLE
    patch = frame[int(height * y0):int(height * y1), int(width * x0):int(width * x1)]
    flat = patch.reshape(-1, 3).astype(np.int32)

    quantized = flat // 8
    keys = quantized[:, 0] * 4096 + quantized[:, 1] * 64 + quantized[:, 2]
    values, counts = np.unique(keys, return_counts=True)
    best = values[counts.argmax()]
    return flat[keys == best].mean(axis=0)


def _find_columns(frame: np.ndarray, grid: Rect) -> list[Rect]:
    """按卡片底色找出每一列的横向范围。"""
    scale = _ui_scale(frame)
    min_width = max(20, round(CARD_MIN_WIDTH * scale))
    left_inset, right_inset = (round(v * scale) for v in CARD_INSET_X)

    bg = _card_bg_color(frame)
    region = frame[grid.to_slice()].astype(np.int32)
    mask = (np.abs(region - bg).sum(axis=2) < CARD_BG_TOL).astype(np.uint8)

    # 先圈出卡片正文所在的那几行，列投影只在这些行上做。卡片上下的背景色有时和
    # 卡片底色很接近，一起算进来的话没卡的地方也会投出很高的列（实测一屏只有一行
    # 卡时，卡片下方的大片背景让中间列的计数虚高到能过阈值，左右两列却被刷掉）。
    row_counts = mask.sum(axis=1)
    peak = int(row_counts.max()) if row_counts.size else 0
    if peak < grid.w * CARD_ROW_MIN_FILL:
        return []
    body = mask[row_counts >= peak * CARD_ROW_FILL]
    if body.shape[0] == 0:
        return []

    counts = body.sum(axis=0)
    bands = _bands(
        counts,
        min_len=min_width,
        min_pixels=int(body.shape[0] * CARD_COL_FILL),
        merge_gap=max(2, round(4 * scale)),
    )

    columns: list[Rect] = []
    for start, end in bands:
        x0 = grid.x + start + left_inset
        x1 = grid.x + end - right_inset
        if x1 - x0 >= min_width:
            columns.append(Rect(x=x0, y=grid.y, w=x1 - x0, h=grid.h))
    return columns


def _cards_in_column(frame: np.ndarray, column: Rect, threshold: int) -> list[Card]:
    """在一列的 x 范围内按行投影切卡片。

    卡片头部「御魂图标 + Lv.0 » Lv.15」的行高是普通属性行的 3~4 倍，拿它当分隔符。
    相邻两个头部起点的间距就是卡片行距，用它界定最后一张卡片——否则最后一张会被
    下面那张只露出一角的卡片头部污染。
    """
    scale = _ui_scale(frame)
    pad = max(1, round(3 * scale))

    mask = _bright_mask(frame[column.to_slice()], threshold)
    counts = mask.sum(axis=1)
    bands = _row_bands(
        mask,
        min_height=max(3, round(6 * scale)),
        min_pixels=max(1, round(3 * scale)),
        merge_gap=max(1, round(3 * scale)),
    )

    # 滤掉底部边框/分隔线这类「暗亮曲线」段，否则没显示固有属性的卡片会把边框
    # 当成一行属性，读出一行空文本。
    min_peak = column.w * CARD_ROW_PEAK_RATIO
    bands = [b for b in bands if int(counts[b[0]:b[1]].max()) >= min_peak]
    if not bands:
        return []

    heights = [end - start for start, end in bands]
    median = float(np.median(heights))
    if median <= 0:
        return []

    headers = [i for i, h in enumerate(heights) if h > ICON_BAND_RATIO * median]
    if not headers:
        return []

    starts = [bands[i][0] for i in headers]
    pitch = int(np.median(np.diff(starts))) if len(starts) >= 2 else int(median * 15)
    bounds = starts + [starts[-1] + pitch]

    cards: list[Card] = []
    for index, header_index in enumerate(headers):
        limit = bounds[index + 1]
        band = bands[header_index]
        header = Rect(
            x=column.x, y=column.y + band[0], w=column.w, h=band[1] - band[0]
        )
        rows: list[TextRow] = []
        for start, end in bands[header_index + 1:]:
            if start >= limit:
                break
            y0 = max(column.y + start - pad, 0)
            y1 = min(column.y + end + pad, frame.shape[0])
            rows.append(
                TextRow(
                    rect=Rect(x=column.x, y=y0, w=column.w, h=y1 - y0),
                    kind="main" if not rows else "sub",
                )
            )
        if rows:
            top = rows[0].rect.y
            bottom = rows[-1].rect.y1
            cards.append(
                Card(
                    rect=Rect(x=column.x, y=top, w=column.w, h=bottom - top),
                    rows=tuple(rows),
                    header=header,
                    row=index,
                )
            )
    return cards


def find_cards(frame: np.ndarray, threshold: int = 100) -> list[Card]:
    """批量强化界面：返回网格里所有能读到的御魂卡片（按列优先顺序）。

    底部只露出一角的卡片因为读不到属性行，会被自然跳过。
    """
    grid = grid_roi(frame)
    cards: list[Card] = []
    for column_index, column in enumerate(_find_columns(frame, grid)):
        for card in _cards_in_column(frame, column, threshold):
            cards.append(
                Card(
                    rect=card.rect,
                    rows=card.rows,
                    header=card.header,
                    column=column_index,
                    row=card.row,
                )
            )
    return cards


def find_panel_card(
    frame: np.ndarray,
    roi: Rect | None = None,
    threshold: int = 100,
) -> Card | None:
    """只剩一张御魂时，用面板 ROI 的行投影反推卡片框。

    _find_columns 靠「卡片底色」找列，画面背景与卡片底色接近时整列都会被并进
    背景掩码，于是一张卡也找不到。而剩余的那张卡总摆在网格左上角第一格，
    PANEL_ROI 正好覆盖它的正文宽度，所以直接在这个 ROI 里做行投影：首个远高于
    其余行的高段就是头部「套装图标 + Lv.0 » Lv.15」，其余是属性行。

    这样构造出的 Card 和 find_cards 的结构一致，徽章读取（依赖 rect.x1）与头部
    图标识别都能照常走。
    """
    roi = roi or panel_roi(frame)
    rows = find_rows(frame, roi=roi, threshold=threshold, drop_icon=False)
    if len(rows) < 2:
        return None

    header, body = rows[0], rows[1:]
    median = float(np.median([row.rect.h for row in body]))
    if median <= 0 or header.rect.h <= ICON_BAND_RATIO * median:
        return None

    body_rows = tuple(
        TextRow(rect=row.rect, kind="main" if index == 0 else "sub")
        for index, row in enumerate(body)
    )
    top = body_rows[0].rect.y
    bottom = body_rows[-1].rect.y1
    return Card(
        rect=Rect(x=roi.x, y=top, w=roi.w, h=bottom - top),
        rows=body_rows,
        header=header.rect,
    )
