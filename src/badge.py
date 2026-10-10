"""读批量强化卡片上「强化次数」徽章里的数字。

每张卡片里，被强化过的属性行右侧边框上会挂一个浅蓝圆盘，圆盘里是这条属性本次
被强化了几次。位置固定在卡片右边界附近、与该行垂直居中；数字是纯白，比圆盘和
卡片底色都亮得多，所以先用亮度把数字抠出来，再按质心裁一小块放大识别。

分配方式是「先找簇、再就近归行」：整条右带一次性找出所有白色数字簇，每个簇归给
垂直距离最近的行。早先按「行中心 ± 0.75 行高」逐行搜，相邻行的徽章会落进彼此的
搜索窗口——行高 20px、行距也是 20px，窗口必然重叠，于是出现漏读和串行。

实测（1402x790 客户区）：数字高约 8px、宽约 6px，8 倍放大后 RapidOCR 置信度 0.99。
"""

from __future__ import annotations

import cv2
import numpy as np

from src.ocr import _engine
from src.panel import BASE_HEIGHT, Card, Rect

# 搜索窗口：相对卡片右边界向右扩展的范围（基准分辨率 790 高）
SEARCH_X = (1, 24)
# 搜索窗口在卡片上下各外扩多少，避免首尾行的数字被切掉
SEARCH_PAD_Y = 5

# 数字是纯白（实测 BGR 约 240,231,211），圆盘（约 155,100,73）和卡片底色都远低于它。
# 实测非徽章区域的灰度峰值只有 140~160，190 这个阈值能把文字、边框全挡掉。
DIGIT_GRAY = 190

# 连通域最小面积（基准分辨率）：实测单个数字约 20~53 个白像素，取 8 滤掉噪点
MIN_AREA = 8

# 数字簇与行中心的垂直距离超过行高的这个比例就不认（避免把边框误挂到某行）
ASSIGN_TOL_RATIO = 0.6

# 以数字质心为中心的裁剪半宽/半高（基准分辨率）
BOX_HALF = (13, 15)

UPSCALE = 8

# 数字画在圆盘里，裁得松一点 OCR 就会把「圆盘 + 数字」整体读成圈码（① ② …）。
# 详情面板上实测裁 22x26 有 2 个徽章读成 ①、裁圆盘本身（16x16）则四个都对，
# 所以那边改成贴着圆盘裁（见 src/detail.py）。这里顺手认一下圈码，省得漏读，
# 也避免 int('①') 抛异常。
_CIRCLED_DIGITS = {"①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5}


def _scale(frame: np.ndarray) -> float:
    return frame.shape[0] / BASE_HEIGHT


def _digit_clusters(frame: np.ndarray, card: Card) -> list[tuple[int, int, int]]:
    """在卡片右侧带里找出所有白色数字簇，返回 [(质心x, 质心y, 白像素数)]。

    用连通域而不是行投影：数字「1」只有 4px 宽、笔画细，抗锯齿会把一个数字切成
    上下几段，行投影按「每行至少几个白像素」筛的时候会把这些段整段丢掉。
    """
    scale = _scale(frame)
    x0 = card.rect.x1 + round(SEARCH_X[0] * scale)
    x1 = card.rect.x1 + round(SEARCH_X[1] * scale)
    pad = round(SEARCH_PAD_Y * scale)
    y0 = max(card.rect.y - pad, 0)
    y1 = min(card.rect.y1 + pad, frame.shape[0])
    if x1 > frame.shape[1] or x1 <= x0 or y1 <= y0:
        return []

    gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    mask = (gray >= DIGIT_GRAY).astype(np.uint8)

    # 笔画上的抗锯齿断点先黏回去，否则一个数字会算成好几块
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(2, round(MIN_AREA * scale * scale))

    clusters: list[tuple[int, int, int]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cx, cy = centroids[label]
        clusters.append((x0 + int(round(cx)), y0 + int(round(cy)), area))
    return clusters


def read_digit_in_box(frame: np.ndarray, box: Rect) -> int | None:
    """把 box 这一小块放大后 OCR，读出里面的强化次数；读不到返回 None。

    位置已经由调用方定好，所以跳过检测直接识别：数字「1」只有 4px 宽，检测器
    经常整块漏掉，实测同一批徽章里检测模式 9/12、只识别模式 12/12。
    """
    patch = frame[max(box.y, 0):box.y1, max(box.x, 0):box.x1]
    if patch.size == 0:
        return None
    big = cv2.resize(patch, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)

    result, _ = _engine()(big, use_det=False, use_cls=False, use_rec=True)
    if not result:
        return None
    if isinstance(result[0], (list, tuple)):
        text = "".join(str(item[0]) for item in result)
    else:
        text = str(result)
    for ch in text:
        if ch in _CIRCLED_DIGITS:
            value = _CIRCLED_DIGITS[ch]
        elif ch.isdecimal():
            value = int(ch)
        else:
            continue
        # 徽章只可能是 1~5（最多强化 5 次）
        return value if 1 <= value <= 5 else None
    return None


def _read_digit(frame: np.ndarray, center: tuple[int, int]) -> int | None:
    """裁出数字质心周围一小块，放大后 OCR。"""
    scale = _scale(frame)
    half_w, half_h = (round(v * scale) for v in BOX_HALF)
    cx, cy = center
    return read_digit_in_box(
        frame, Rect(x=cx - half_w, y=cy - half_h, w=2 * half_w, h=2 * half_h)
    )


def digit_center(frame: np.ndarray, card: Card, row_index: int) -> tuple[int, int] | None:
    """该行徽章里数字的质心；没有徽章返回 None。"""
    row = card.rows[row_index]
    scale = _scale(frame)
    tolerance = ASSIGN_TOL_RATIO * row.rect.h
    center_y = (row.rect.y + row.rect.y1) / 2

    best: tuple[int, int] | None = None
    best_distance = tolerance
    for cx, cy, _ in _digit_clusters(frame, card):
        distance = abs(cy - center_y)
        if distance < best_distance:
            best, best_distance = (cx, cy), distance
    return best


def read_badge(frame: np.ndarray, card: Card, row_index: int) -> int | None:
    """读出该行的强化次数。没有徽章返回 None，识别不出数字返回 None。"""
    center = digit_center(frame, card, row_index)
    return _read_digit(frame, center) if center is not None else None


def read_badges(frame: np.ndarray, card: Card) -> dict[int, int]:
    """读一张卡片所有行的强化次数，返回 {行下标: 次数}。只保留读到数字的行。

    一个数字簇只归给垂直距离最近的一行；同一行若有多个簇（数字被切开），取白像素
    最多的那个。
    """
    clusters = _digit_clusters(frame, card)
    if not clusters:
        return {}

    row_centers = [(row.rect.y + row.rect.y1) / 2 for row in card.rows]
    heights = [row.rect.h for row in card.rows]
    tolerance = ASSIGN_TOL_RATIO * float(np.median(heights))

    best_per_row: dict[int, tuple[int, int, int]] = {}
    for cluster in clusters:
        cx, cy, count = cluster
        index, distance = min(
            ((i, abs(cy - center)) for i, center in enumerate(row_centers)),
            key=lambda pair: pair[1],
        )
        if distance > tolerance:
            continue
        if index not in best_per_row or count > best_per_row[index][2]:
            best_per_row[index] = cluster

    result: dict[int, int] = {}
    for index, (cx, cy, _) in best_per_row.items():
        value = _read_digit(frame, (cx, cy))
        if value is not None:
            result[index] = value
    return result
