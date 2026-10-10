"""御魂强化面板的文字识别。

用 RapidOCR（PP-OCRv4，onnxruntime 后端）识别 panel.find_rows 切出的每一条文字行。
一条行就是一条属性，形如「防御加成 2.75% » 11.02%」，识别后交给 parse 拆成属性名与最终值。

识别前把裁剪条放大：面板正文只有 20px 上下，直接喂给 OCR 中文容易糊，
放大到 2~3 倍后模型才有足够的笔画细节。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from src.panel import Rect, TextRow, crop


@dataclass(frozen=True)
class OcrLine:
    """一行的识别结果。text 是该行所有文本框按 x 排序拼接后的整串。"""

    rect: Rect
    text: str
    score: float
    boxes: tuple[tuple[float, float, float, float], ...] = ()


@lru_cache(maxsize=1)
def _engine():
    """RapidOCR 初始化要加载 3 个 onnx 模型，只做一次。"""
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def _upscale(img: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def read_row(
    frame: np.ndarray,
    row: TextRow,
    scale: int = 2,
    min_score: float = 0.3,
) -> OcrLine:
    """识别单行。按文本框 x 中心排序后拼接，避免把左右两段读反。"""
    img = _upscale(crop(frame, row.rect, scale=1), scale)
    result, _ = _engine()(img)

    if not result:
        return OcrLine(rect=row.rect, text="", score=0.0)

    items: list[tuple[float, float, str, float, tuple[float, float, float, float]]] = []
    for box, text, score in result:
        if score < min_score:
            continue
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        center_x = sum(xs) / len(xs)
        x0, x1 = min(xs) / scale, max(xs) / scale
        y0, y1 = min(ys) / scale, max(ys) / scale
        items.append((center_x, y0, text.strip(), float(score), (x0, y0, x1, y1)))

    items.sort(key=lambda item: item[0])

    text = "".join(item[2] for item in items)
    score = min((item[3] for item in items), default=0.0)
    boxes = tuple(item[4] for item in items)
    return OcrLine(rect=row.rect, text=text, score=score, boxes=boxes)


def read_rows(
    frame: np.ndarray,
    rows: list[TextRow],
    scale: int = 2,
    min_score: float = 0.3,
) -> list[OcrLine]:
    return [read_row(frame, row, scale=scale, min_score=min_score) for row in rows]


def read_region(
    frame: np.ndarray,
    region: Rect,
    rows: list[TextRow],
    scale: int = 2,
    min_score: float = 0.3,
) -> list[OcrLine]:
    """整块区域只跑一次 OCR，再把识别框按 y 归到各行。

    批量强化一屏有几十行，逐行调用 OCR 光模型调度开销就要几十秒；整块识别一次，
    再把每个文本框按中心 y 落到对应的行里，速度快一个数量级。
    """
    img = _upscale(crop(frame, region, scale=1), scale)
    result, _ = _engine()(img)

    buckets: list[list[tuple[float, str, float]]] = [[] for _ in rows]

    def _nearest(center_y: float) -> int:
        for index, row in enumerate(rows):
            if row.rect.y - 4 <= center_y <= row.rect.y1 + 4:
                return index
        return min(
            range(len(rows)),
            key=lambda i: abs((rows[i].rect.y + rows[i].rect.y1) / 2 - center_y),
        )

    for box, text, score in result or []:
        if score < min_score:
            continue
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        center_x = sum(xs) / len(xs) / scale + region.x
        center_y = sum(ys) / len(ys) / scale + region.y
        buckets[_nearest(center_y)].append((center_x, text.strip(), float(score)))

    lines: list[OcrLine] = []
    for index, row in enumerate(rows):
        items = sorted(buckets[index], key=lambda item: item[0])
        lines.append(
            OcrLine(
                rect=row.rect,
                text="".join(item[1] for item in items),
                score=min((item[2] for item in items), default=0.0),
            )
        )
    return lines


def read_boxes(
    frame: np.ndarray,
    region: Rect,
    scale: int = 2,
    min_score: float = 0.3,
) -> list[OcrLine]:
    """整块区域只跑一次 OCR，每个文本框单独成一行（不按行归并）。

    归行的活儿反过来交给调用方。御魂详情面板的底色是浅蓝、文字是白/深蓝，对比度
    极低，按亮度做行投影切不出行（见 src/detail.py），只能先整块识别、再按文本框
    的 y 把结果归成行。
    """
    img = _upscale(crop(frame, region, scale=1), scale)
    result, _ = _engine()(img)

    lines: list[OcrLine] = []
    for box, text, score in result or []:
        if score < min_score:
            continue
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        x0, x1 = min(xs) / scale + region.x, max(xs) / scale + region.x
        y0, y1 = min(ys) / scale + region.y, max(ys) / scale + region.y
        lines.append(
            OcrLine(
                rect=Rect(
                    x=round(x0), y=round(y0), w=round(x1 - x0), h=round(y1 - y0)
                ),
                text=text.strip(),
                score=float(score),
            )
        )
    return lines


def annotate(frame: np.ndarray, lines: list[OcrLine], color=(0, 255, 0)) -> np.ndarray:
    """把识别结果画回图上，用于人工核对。"""
    canvas = frame.copy()
    for line in lines:
        rect = line.rect
        cv2.rectangle(canvas, (rect.x, rect.y), (rect.x1, rect.y1), color, 1)
        cv2.putText(
            canvas,
            line.text,
            (rect.x, max(rect.y - 3, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas
