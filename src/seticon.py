"""套装图标识别：在卡片头部左侧的圆形图标上做多尺度模板匹配。

图标素材是 data/soul_icons/<中文套装名>.png（80x80 圆形、四角透明），从平安志
安装包的内嵌资源里解出。游戏会把同一套美术按御魂状态重新染色（同一张图整体偏蓝
或偏青），所以只比灰度结构、不比颜色。

匹配走归一化互相关（TM_CCOEFF_NORMED）：把模板缩放到若干候选直径、按 alpha 盖到
卡片底色上（与游戏合成方式一致），再在「卡片头部左下角」一小块窗口里逐位置滑窗
比对；每个套装取自己的最高分，最高分者即识别结果。除了绝对分数还要看与第二名的
差距，两者都不够就不下结论——宁可返回 None，让上层退回「乐观上界」。

标定基准：阴阳师 PC 端客户区 1402x790，此时图标直径约 59 像素。

模板匹配的峰值位置同时就是图标本身的位置，所以顺带也能拿到图标的外接方框，
号位识别就建在这个方框上：图标背后的六边形底盘有一个角/尖被高亮成浅蓝，方位
固定对应号位（1 左上 / 2 左 / 3 左下 / 4 右下 / 5 右 / 6 右上）。只统计圆外那
一圈，免得把头像美术里的蓝当成角标；六个方位各取一个扇形做计数，最优者即号位。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .panel import Card, Rect, _ui_scale

ICON_DIR = Path(__file__).resolve().parents[1] / "data" / "soul_icons"

# 客户区 790 高时套装图标的直径（像素）
BASE_DIAM = 59

# 候选直径相对基准的倍数，覆盖 1~2 像素的缩放误差
DIAM_STEPS = (0.94, 0.97, 1.0, 1.03, 1.06)

# 搜索窗尺寸（相对客户区高度换算）：贴卡片左边缘，底边对齐头部行底边。
# 图标底边与头部行底边对齐，顶部留白由行的合并情况决定，所以从底边往上取。
WINDOW_W = 120
WINDOW_H = 88
WINDOW_BOTTOM_GAP = 4

# 判定的两道门槛：绝对分数，以及与第二名的差距
MIN_SCORE = 0.55
MIN_MARGIN = 0.08

# 网格检测不到时（只剩一张卡）用的兜底窗口：批量强化里剩下的卡总是从左上角那张
# 开始摆，所以第一张卡的头部图标在客户区里的相对位置是固定的。
# 标定：1402x790 下图标实际落在 x 16.8%~21.4%、y 11.9%~20.0%，这里再放一点余量。
FIRST_ICON_ROI = (0.150, 0.085, 0.27, 0.25)


# ---- 号位角标 ----

# 图标背后的六边形底盘上有一个角/尖被高亮成浅蓝，高亮方位固定对应号位。
# 单位向量是图像坐标（x 向右、y 向下），下标 +1 就是号位。
SLOT_DIRECTIONS = (
    (-0.707, -0.707),   # 1 号位：左上
    (-1.0, 0.0),        # 2 号位：左
    (-0.707, 0.707),    # 3 号位：左下
    (0.707, 0.707),     # 4 号位：右下
    (1.0, 0.0),         # 5 号位：右
    (0.707, -0.707),    # 6 号位：右上
)

# 只统计圆外这一圈：圆内是头像美术，蓝色会被误判成角标
SLOT_RING_MIN = 0.98
SLOT_RING_MAX = 1.60

# 角标的浅蓝（实测 BGR 约 225,150,75）：B 明显高于 R，且整体足够亮
SLOT_B_MIN = 150
SLOT_B_MINUS_R = 60

# 扇形半角 22.5°，六个方位正好铺满一圈
SLOT_COS_HALF = float(np.cos(np.deg2rad(22.5)))

# 判定门槛：最优扇形的像素数要达到图标面积的这个比例，且至少是次优扇形的这么多倍。
# 实测 6 张卡的最优/次优比值都在 2.6 以上、占比都在 0.055 以上，这里留了余量。
SLOT_MIN_FRACTION = 0.025
SLOT_MIN_RATIO = 1.5

# 模板匹配分数低于这个值就不信它给出的位置：宁可判不出号位，也不要判错
LOCATE_MIN_SCORE = 0.55


@dataclass(frozen=True)
class SetMatch:
    name: str
    score: float
    margin: float
    source: str = "icon"    # 识别途径：icon 图标模板匹配 / text 面板上的套装名文字


def set_names(icon_dir: Path | None = None) -> tuple[str, ...]:
    """所有已知套装名，就是 data/soul_icons 里模板的文件名。

    规则文件里的 setIds 用的也是同一批中文名，所以御魂详情页直接按文本比对时
    可以拿它当名录。
    """
    return tuple(sorted(path.stem for path in (icon_dir or ICON_DIR).glob("*.png")))


def _load_icons(icon_dir: Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """读入模板，返回 (套装名, 灰度图, alpha)，后两者都是 float32 的 [0,1]/[0,255]。"""
    icons: list[tuple[str, np.ndarray, np.ndarray]] = []
    for path in sorted(icon_dir.glob("*.png")):
        raw = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if raw is None:
            continue
        if raw.ndim == 2:
            raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGRA)
        elif raw.shape[2] == 3:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2BGRA)
        gray = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
        alpha = raw[:, :, 3].astype(np.float32) / 255.0
        icons.append((path.stem, gray, alpha))
    return icons


def _gray_of(bgr: np.ndarray) -> float:
    b, g, r = (float(v) for v in bgr)
    return 0.114 * b + 0.587 * g + 0.299 * r


class SetMatcher:
    """同一批卡片共用一份预缩放模板（按帧高缓存）。"""

    def __init__(self, icon_dir: Path | None = None) -> None:
        self._icons = _load_icons(icon_dir or ICON_DIR)
        self._cache: dict[int, list[tuple[str, np.ndarray, np.ndarray]]] = {}

    def _templates(self, frame: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """把模板缩放成「乘过 alpha 的灰度贡献 ga」和「透明权重 inv」。

        合成后的灰度是线性的：gray(art*a + bg*(1-a)) = a*gray(art) + (1-a)*gray(bg)，
        所以 ga = a*gray(art) 与 inv = 1-a 可以预先算好，底色随卡片现取。
        """
        key = round(BASE_DIAM * _ui_scale(frame))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        built: list[tuple[str, np.ndarray, np.ndarray]] = []
        for step in DIAM_STEPS:
            diam = max(8, round(key * step))
            for name, gray, alpha in self._icons:
                a = cv2.resize(alpha, (diam, diam), interpolation=cv2.INTER_AREA)
                g = cv2.resize(gray, (diam, diam), interpolation=cv2.INTER_AREA)
                built.append((name, a * g, 1.0 - a))
        self._cache[key] = built
        return built

    def _window(self, frame: np.ndarray, card: Card) -> Rect | None:
        scale = _ui_scale(frame)
        height, width = frame.shape[:2]
        x0 = max(card.header.x, 0)
        y1 = min(card.header.y1 + round(WINDOW_BOTTOM_GAP * scale), height)
        y0 = max(y1 - round(WINDOW_H * scale), 0)
        x1 = min(x0 + round(WINDOW_W * scale), width)
        if x1 - x0 < 16 or y1 - y0 < 16:
            return None
        return Rect(x=x0, y=y0, w=x1 - x0, h=y1 - y0)

    def _rank(
        self, frame: np.ndarray, region: np.ndarray, origin: tuple[int, int]
    ) -> list[tuple[str, float, Rect]]:
        """在 region 里给每个套装打分（滑窗最高分），从高到低排。

        分数的峰值位置就是图标在 region 里的位置，一并换算成整帧坐标下的外接方框
        返回——号位识别直接拿它当圆心和半径，不用再定位一次。
        """
        bg_gray = _gray_of(np.median(region.reshape(-1, 3), axis=0))
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).astype(np.float32)

        best: dict[str, tuple[float, Rect]] = {}
        for name, ga, inv in self._templates(frame):
            if ga.shape[0] > gray.shape[0] or ga.shape[1] > gray.shape[1]:
                continue
            result = cv2.matchTemplate(gray, ga + bg_gray * inv, cv2.TM_CCOEFF_NORMED)
            _, top, _, loc = cv2.minMaxLoc(result)
            if top > best.get(name, (-2.0, None))[0]:
                diam = ga.shape[0]
                best[name] = (
                    float(top),
                    Rect(x=origin[0] + loc[0], y=origin[1] + loc[1], w=diam, h=diam),
                )
        return sorted(
            ((name, score, rect) for name, (score, rect) in best.items()),
            key=lambda item: -item[1],
        )

    @staticmethod
    def _decide(ranked: list[tuple[str, float, Rect]]) -> SetMatch | None:
        if len(ranked) < 2:
            return None
        score = ranked[0][1]
        margin = score - ranked[1][1]
        if score < MIN_SCORE or margin < MIN_MARGIN:
            return None
        return SetMatch(name=ranked[0][0], score=score, margin=margin)

    def analyze(self, frame: np.ndarray, card: Card) -> tuple[SetMatch | None, Rect | None]:
        """一次匹配同时给出套装识别结果和图标外接方框（号位识别要用）。

        方框只在匹配分数够高时才给：位置偏了会把号位一起带偏，宁可返回 None。
        """
        window = self._window(frame, card)
        if window is None or not self._icons:
            return None, None
        ranked = self._rank(frame, frame[window.to_slice()], (window.x, window.y))
        match = self._decide(ranked)
        icon = ranked[0][2] if ranked and ranked[0][1] >= LOCATE_MIN_SCORE else None
        return match, icon

    def search(self, frame: np.ndarray, roi: tuple[float, float, float, float] | None = None) -> SetMatch | None:
        """没有卡片信息时，在左上角第一张卡的位置找图标（只剩一张卡时用）。"""
        if not self._icons:
            return None
        x0, y0, x1, y1 = roi or FIRST_ICON_ROI
        height, width = frame.shape[:2]
        left, top = int(width * x0), int(height * y0)
        region = frame[top:int(height * y1), left:int(width * x1)]
        if region.size == 0:
            return None
        return self._decide(self._rank(frame, region, (left, top)))


# 模块级复用的默认实例，省得调用方自己管缓存
_DEFAULT_MATCHER: SetMatcher | None = None


def _matcher() -> SetMatcher:
    global _DEFAULT_MATCHER
    if _DEFAULT_MATCHER is None:
        _DEFAULT_MATCHER = SetMatcher()
    return _DEFAULT_MATCHER


def analyze(frame: np.ndarray, card: Card) -> tuple[SetMatch | None, Rect | None]:
    """一次匹配同时给出套装识别结果和图标外接方框（号位识别要用）。"""
    return _matcher().analyze(frame, card)


def search(frame: np.ndarray, roi: tuple[float, float, float, float] | None = None) -> SetMatch | None:
    """没有卡片信息时，在左上角区域直接搜套装图标。"""
    return _matcher().search(frame, roi)


def slot_from_icon(frame: np.ndarray, icon: Rect) -> int | None:
    """从图标背后的高亮角标判断号位（1~6）；判不出来返回 None。

    icon 是图标的外接方框（见 SetMatcher.analyze）。六边形底盘被高亮的那个角/尖
    落在圆外，先按颜色把圆外一圈的浅蓝抠出来，再按方位做扇形计数；最优扇形还得
    显著高于其余五个方位，否则不下结论。
    """
    radius = icon.w / 2
    if radius < 8:
        return None
    cx = icon.x + radius
    cy = icon.y + radius

    pad = int(round(radius * SLOT_RING_MAX))
    x0, y0 = int(round(cx)) - pad, int(round(cy)) - pad
    x1, y1 = x0 + 2 * pad, y0 + 2 * pad
    if x0 < 0 or y0 < 0 or x1 > frame.shape[1] or y1 > frame.shape[0]:
        return None

    patch = frame[y0:y1, x0:x1]
    rows, cols = np.mgrid[y0:y1, x0:x1]
    dx = (cols - cx).astype(np.float64)
    dy = (rows - cy).astype(np.float64)
    dist = np.sqrt(dx * dx + dy * dy)

    blue, green, red = (patch[:, :, k].astype(np.int32) for k in range(3))
    mask = (
        (blue - red > SLOT_B_MINUS_R)
        & (blue > SLOT_B_MIN)
        & (blue > green)
        & (dist > SLOT_RING_MIN * radius)
        & (dist <= SLOT_RING_MAX * radius)
    )
    if not mask.any():
        return None

    unit = np.maximum(dist, 1e-6)
    ux, uy = dx / unit, dy / unit
    counts = [
        int(np.count_nonzero(mask & (ux * vx + uy * vy >= SLOT_COS_HALF)))
        for vx, vy in SLOT_DIRECTIONS
    ]
    best = max(range(6), key=lambda index: counts[index])
    second = max(counts[index] for index in range(6) if index != best)
    if counts[best] < SLOT_MIN_FRACTION * (2 * radius) ** 2:
        return None
    if counts[best] < SLOT_MIN_RATIO * second:
        return None
    return best + 1
