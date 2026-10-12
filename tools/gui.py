"""御魂评分器的图形界面。

从截图到评分一条命令跑完：选窗口（或打开一张截图）-> 点「开始分析」-> 左边看截图、
右边看每件御魂的分数和建议，点某一件在下面看属性明细和评分依据，需要时导出
JSON / CSV / 标注图。

识别逻辑全在 src/pipeline.py，这里只做「取图 + 展示 + 导出」。分析放在子线程里跑
（一屏六张卡要 5 秒上下），跑完用 after() 回主线程刷界面——Tk 的控件只能在主线程碰。

界面走 Apple HIG 的卡片风格：整窗一张柔和渐变背景，上面浮着几张「毛玻璃」卡片。
注意 tkinter 没有系统级亚克力模糊（原生只有纯色 / 位图贴图），这里的毛玻璃是「手工
合成」出来的：取卡片所在位置的背景渐变 -> 高斯模糊 -> 按比例混白 -> 用抗锯齿圆角蒙版
裁切成圆角，再补投影、外发光、内高光和外发丝边。因此卡片四周的投影是叠在真实渐变上
的，角落能无缝融进背景。

用法:
    python tools/gui.py
"""

from __future__ import annotations

import csv
import ctypes
import json
import sys
import threading
import time
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

# 打包（PyInstaller）后资源被解到 sys._MEIPASS，源码运行时就是仓库根目录
ROOT = Path(getattr(sys, "_MEIPASS", None) or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageTk

from src.ocr import annotate
from src.parse import format_value, unparsed
from src.paths import writable_root
from src.pipeline import (
    MODE_BATCH,
    MODE_SINGLE_CARD,
    MODE_SOUL_DETAIL,
    AnalysisError,
    PanelAnalysis,
    SoulReport,
    analyze_frame,
)
from src.score import ATTR_ID_TO_NAME, load_rules
from src.win_capture import capture_client_area, enable_dpi_awareness, list_windows

APP_TITLE = "御魂评分器"
# 作者信息：界面顶栏署名 + 「项目主页」按钮直达这里；改仓库地址时只改这一处
AUTHOR = "阳小楊"
YYS_ACCOUNT = "鲛人之歌 · 阳小楊"
REPO_URL = "https://github.com/asta0511/yys-soul-scorer"
# 导出产物写到 exe 旁边（打包后不能写进只读的 _MEIPASS 临时目录）
OUT_DIR = writable_root() / "output" / "panel_read"

# 与命令行保持一致：不改这两个参数，界面和 CLI 的结果就能对上
THRESHOLD = 100
SCALE = 2

KIND_LABEL = {"main": "主属性", "sub": "副属性", "innate": "固有"}
VERDICT_TAG = {"留": "keep", "观察": "observe", "弃": "discard"}

# 视觉令牌：Apple HIG 风。主 / 次文字对卡片填充色 GLASS_FILL(#F8F9FD) 都在 AA 4.5:1 以上，
# 主按钮白字对 accent 4.7:1 也达 AA。改色前先按对比度复算。
COLOR = {
    "fg": "#1D1D1F",            # 主文字
    "secondary": "#6E6E73",     # 次要文字（对 GLASS_FILL 约 4.8:1，达 AA）
    "tertiary": "#8E8E93",
    "separator": "#E5E5EA",
    "glass": "#F8F9FD",         # 卡片内部填充 —— 所有 ttk 控件的背景都用它
    "well": "#EEF1F6",          # 预览区那口「井」的底色
    "accent": "#0071E3",        # 主按钮，白字对比度 4.7:1，达 AA
    "accent_pressed": "#0059B2",
    "accent_hover": "#1A80EE",
    "on_accent": "#FFFFFF",
    "hover": "#EFF3FA",
    "select": "#DCEBFF",
    "control_border": "#DCE0E8",
    "disabled_fg": "#AEAEB2",
}
# ttk 控件的背景色只能是一个固定值，单独抽出来复用；它就是「玻璃」的近似色。
GLASS_FILL = COLOR["glass"]

# 建议只靠底色区分，文字始终是最高对比的 #1D1D1F——不靠颜色单独传达信息。
VERDICT_ROW = {"留": "#E3F6E7", "观察": "#FCF2DC", "弃": "#EFEFEF"}
# 预览是浅底，方框要用在浅色上也够清楚的深色版（非文字元素按 3:1 取）
VERDICT_MARK = {"留": "#1B7F3B", "观察": "#9A5B00", "弃": "#6E6E73"}
# 画在游戏截图（深底）上的 cv2 BGR 值保持不变
VERDICT_BGR = {"留": (60, 170, 60), "观察": (0, 150, 230), "弃": (150, 150, 150)}

MODE_LABEL = {
    MODE_BATCH: "批量强化",
    MODE_SINGLE_CARD: "单个御魂",
    MODE_SOUL_DETAIL: "御魂详情",
}

# 布局令牌：间距全走 8 的倍数，16 优先（HIG 的 8pt 栅格）
MARGIN = 16
# 卡片间隙 = 2 * GlassCard.BLEED（卡片画布外扩，留给投影/外发光）+ 分隔条宽度。
# 32 = 2*10 + 12，这样分隔条正好落在两张卡片画布之间的空隙里，不压住它们的投影。
GAP = 32
# 卡片高度 = 内容高 + 2*PADDING(16)。顶栏 = 标题行 + 间距 + 工具条，状态栏只有一排按钮。
# 这两个只是 100% 缩放下算出来的兜底值：进程是 DPI 感知的，125% 缩放下同一套字体要多占
# 四分之一的高度，写死就会把按钮文字裁掉，所以运行时改用 _chrome_heights() 按控件实际
# 需求重新算，这两个数只在控件还没测量出来时兜底。
HEADER_H = 112
STATUS_H = 72
# 拖动分隔条时的最小尺寸，防止某一块被拖到看不见
MIN_LEFT_W = 320
MIN_RIGHT_W = 380
MIN_LIST_H = 150
MIN_DETAIL_H = 200

# ---------------------------------------------------------------- 渐变背景

# 斜向（左上 -> 右下）多段色标，柔和、极浅、低饱和
_GRADIENT_STOPS = (
    (0.00, (238, 244, 253)),   # 冷白蓝
    (0.35, (233, 238, 252)),
    (0.62, (240, 235, 250)),   # 淡薰衣草
    (0.85, (233, 245, 243)),   # 淡薄荷
    (1.00, (244, 246, 251)),
)
# 按尺寸缓存，窗口尺寸没变就不重算
_GRADIENT_CACHE: dict[tuple[int, int], Image.Image] = {}
# 圆角蒙版只跟尺寸和半径有关，缓存下来省掉重复的 2 倍尺寸绘制
_MASK_CACHE: dict[tuple[int, int, int], Image.Image] = {}


def _sample_gradient(t: float) -> tuple[int, int, int]:
    """在色标上按位置 t∈[0,1] 线性插值，取一个 RGB。"""
    stops = _GRADIENT_STOPS
    if t <= stops[0][0]:
        return stops[0][1]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t <= t1:
            f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return tuple(int(round(c0[k] + (c1[k] - c0[k]) * f)) for k in range(3))
    return stops[-1][1]


def _gradient_background(width: int, height: int) -> Image.Image:
    """生成整窗渐变背景：128x72 上按对角线插值 -> BICUBIC 放大 -> 高斯模糊柔化。"""
    width = max(1, int(width))
    height = max(1, int(height))
    key = (width, height)
    cached = _GRADIENT_CACHE.get(key)
    if cached is not None:
        return cached

    small_w, small_h = 128, 72
    small = Image.new("RGB", (small_w, small_h))
    pixels = small.load()
    denom = (small_w - 1) + (small_h - 1)
    for y in range(small_h):
        for x in range(small_w):
            pixels[x, y] = _sample_gradient((x + y) / denom if denom else 0.0)

    image = small.filter(ImageFilter.GaussianBlur(radius=2.0))
    # 先在小图上模糊再放大：半径按放大倍数折算，观感等价但比在全尺寸上模糊便宜两个数量级
    image = image.resize((width, height), Image.BICUBIC)
    _GRADIENT_CACHE[key] = image
    if len(_GRADIENT_CACHE) > 3:
        # 拖动窗口会不断产生新尺寸，只留最近三张，别让缓存里堆一堆全窗位图
        for stale in list(_GRADIENT_CACHE)[:-1]:
            _GRADIENT_CACHE.pop(stale, None)
    return image


class GlassCard(tk.Canvas):
    """一张自己给自己画「毛玻璃底」的卡片。

    卡片内部内容放进 ``card.body``（一个 Frame），外部代码照常用 pack 往里塞控件。
    画法见模块顶部说明：取当前位置的背景渐变 -> 投影 / 外发光 -> 模糊混白的玻璃填充 ->
    内高光 / 外发丝边。所有绘制都包在 try/except 里，失败就退回纯色，绝不让主循环挂掉。
    """

    RADIUS = 18
    PADDING = 16
    # 卡片画布比玻璃本体每边大出 BLEED，专门留给投影和外发光——否则它们会被画布边缘裁掉
    BLEED = 10

    def __init__(self, master: tk.Misc, **kwargs) -> None:
        # 卡片画布本身也用玻璃色，万一图片没画出来也不会有突兀的白块
        super().__init__(master, background=GLASS_FILL, highlightthickness=0, bd=0, **kwargs)
        self._redraw_job: str | None = None
        self._last_geom = (0, 0, 0, 0)
        self._image_id: int | None = None
        self._photo: ImageTk.PhotoImage | None = None

        self.body = tk.Frame(self, background=GLASS_FILL)
        self._window_id = self.create_window(
            self.BLEED + self.PADDING, self.BLEED + self.PADDING, anchor="nw", window=self.body
        )
        self.bind("<Configure>", self._on_configure)

    # -- 事件

    def _on_configure(self, event) -> None:
        w, h = int(event.width), int(event.height)
        if w <= 1 or h <= 1:
            return
        geom = (w, h, int(event.x), int(event.y))
        last = self._last_geom
        # 尺寸与上次相差 < 2px 且位置没变就跳过，避免拖动窗口时狂重绘
        if abs(w - last[0]) < 2 and abs(h - last[1]) < 2 and geom[2:] == last[2:]:
            return
        self._last_geom = geom
        if self._redraw_job is not None:
            self.after_cancel(self._redraw_job)
        self._redraw_job = self.after(50, self._redraw)

    # -- 绘制

    def _redraw(self) -> None:
        self._redraw_job = None
        w, h, x, y = self._last_geom
        if w <= 1 or h <= 1:
            return
        body_w = max(1, w - 2 * (self.BLEED + self.PADDING))
        body_h = max(1, h - 2 * (self.BLEED + self.PADDING))
        try:
            self.itemconfigure(self._window_id, width=body_w, height=body_h)
            self.body.configure(width=body_w, height=body_h)
        except tk.TclError:
            return

        image: Image.Image | None = None
        try:
            image = self._compose(w, h, x, y)
        except Exception:
            image = None  # 任何异常都退回纯色，不能让主循环挂掉

        self.delete("glass")
        if image is None:
            return
        try:
            self._photo = ImageTk.PhotoImage(image)
            self._image_id = self.create_image(0, 0, anchor="nw", image=self._photo, tags="glass")
            # 把玻璃底压到 body 窗口下方，保证控件永远在上面
            self.tag_lower(self._image_id)
        except tk.TclError:
            return

    def _round_mask(self, w: int, h: int) -> Image.Image:
        """抗锯齿圆角蒙版：2 倍尺寸上画圆角矩形再缩回来。按尺寸缓存，不必每次重画。"""
        key = (w, h, self.RADIUS)
        cached = _MASK_CACHE.get(key)
        if cached is not None:
            return cached
        s = 2
        big = Image.new("L", (w * s, h * s), 0)
        ImageDraw.Draw(big).rounded_rectangle(
            [0, 0, w * s - 1, h * s - 1], radius=self.RADIUS * s, fill=255
        )
        mask = big.resize((w, h), Image.LANCZOS)
        if len(_MASK_CACHE) > 24:
            _MASK_CACHE.clear()
        _MASK_CACHE[key] = mask
        return mask

    def _compose(self, w: int, h: int, x: int, y: int) -> Image.Image:
        radius = self.RADIUS
        bleed = self.BLEED
        # 玻璃本体在画布内的位置：四周留出 bleed，投影和外发光就画在这个圈里
        gx = gy = bleed
        gw = max(1, w - 2 * bleed)
        gh = max(1, h - 2 * bleed)

        root = self.winfo_toplevel()
        full_w = max(1, root.winfo_width())
        full_h = max(1, root.winfo_height())

        # 1) 从缓存的整窗背景渐变上裁出卡片所在位置的一块，作为基底
        background = _gradient_background(full_w, full_h)
        cx = max(0, min(x, max(0, background.width - 1)))
        cy = max(0, min(y, max(0, background.height - 1)))
        base = background.crop((cx, cy, cx + w, cy + h)).convert("RGBA")
        if base.size != (w, h):
            base = base.resize((w, h))

        # 2) 投影：玻璃本体下移 4px、外扩 1px 的圆角矩形，纯黑 alpha≈16，模糊 8
        shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(shadow).rounded_rectangle(
            [gx - 1, gy + 3, gx + gw, gy + gh + 4], radius=radius, fill=(0, 0, 0, 16)
        )
        base = Image.alpha_composite(base, shadow.filter(ImageFilter.GaussianBlur(radius=8)))

        # 3) 微光边框（外发光）：沿玻璃本体外扩 1px 的圆角描边，淡蓝 alpha≈70，模糊 5
        glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(glow).rounded_rectangle(
            [gx - 1, gy - 1, gx + gw, gy + gh], radius=radius + 1,
            outline=(156, 194, 255, 70), width=2,
        )
        base = Image.alpha_composite(base, glow.filter(ImageFilter.GaussianBlur(radius=5)))

        # 4) 玻璃填充：玻璃本体范围内的基底模糊 + 按 0.68 混白，再用圆角蒙版裁切
        blurred = background.crop((cx + gx, cy + gy, cx + gx + gw, cy + gy + gh)).convert("RGBA")
        if blurred.size != (gw, gh):
            blurred = blurred.resize((gw, gh))
        blurred = blurred.filter(ImageFilter.GaussianBlur(radius=10))
        glass = Image.blend(blurred, Image.new("RGBA", (gw, gh), (255, 255, 255, 255)), 0.68)
        base.paste(glass, (gx, gy), self._round_mask(gw, gh))

        # 5)+6) 内圈高光 / 最外发丝边：画在独立图层上再合成，避免 ImageDraw 直接把
        #       alpha 写进不透明底图（那样会戳出透明洞）
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rounded_rectangle(
            [gx + 1, gy + 1, gx + gw - 2, gy + gh - 2], radius=radius - 1,
            outline=(255, 255, 255, 90), width=1,
        )
        draw.rounded_rectangle(
            [gx, gy, gx + gw - 1, gy + gh - 1], radius=radius,
            outline=(255, 255, 255, 150), width=1,
        )
        return Image.alpha_composite(base, overlay)


def _to_photo(frame: np.ndarray, max_w: int, max_h: int) -> tuple[ImageTk.PhotoImage, float]:
    """把 BGR 画面缩到能放进预览区，返回 (图片, 缩放系数)。"""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    scale = min(max_w / image.width, max_h / image.height, 1.0)
    if scale < 1.0:
        size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(image), scale


class SoulScorerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.rules = load_rules()
        self.windows = []
        self.frame: np.ndarray | None = None
        self.analysis: PanelAnalysis | None = None
        self.source_title = ""
        self.busy = False

        # 预览图与它上面的卡片方框：分别是 (x0, y0, x1, y1)
        self.photo: ImageTk.PhotoImage | None = None
        self.image_scale = 1.0
        self.card_boxes: list[tuple[float, float, float, float, int]] = []
        self.selected_index = -1
        self._last_canvas_size = (0, 0)
        self.elapsed = 0.0
        self.started_at = 0.0

        # 布局防抖
        self._reflow_job: str | None = None
        self._last_layout_size = (0, 0)
        self._bg_photo: ImageTk.PhotoImage | None = None

        # 可拖动分隔条：左右列宽比例、右列上下行高比例，以及拖动中的临时状态
        self._left_ratio = 0.42
        self._top_ratio = 0.46
        # 顶栏 / 状态栏卡片的实际高度，_apply_layout 每次按控件需求刷新
        self._chrome_h = (HEADER_H, STATUS_H)
        # (方向, 按下时鼠标 x_root, y_root, 起始像素, 可用总长)
        self._drag: tuple[str, float, float, float, float] | None = None
        self._sash_hot = {"v": False, "h": False}
        self._sash_geom: dict[str, tuple[int, int, int, int]] = {}

        self.ui_font = ("Microsoft YaHei UI", 9)
        self.small_font = ("Microsoft YaHei UI", 8)
        self.title_font = ("Microsoft YaHei UI", 16, "bold")
        self.empty_font = ("Microsoft YaHei UI", 11)
        self.section_font = ("Microsoft YaHei UI", 9, "bold")

        root.title(APP_TITLE)
        root.geometry("1280x820")
        root.minsize(960, 620)
        self._init_style()
        self._build_background()
        self._build_cards()
        self._build_sashes()
        self._build_toolbar()
        self._build_status()
        self._build_body()
        self._bind_shortcuts()
        root.bind("<Configure>", self._reflow)
        root.after(50, self._apply_layout)
        self.refresh_windows()

    # ------------------------------------------------------------------ 界面搭建

    def _init_style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        self.root.configure(background=COLOR["well"])

        style.configure(
            ".", background=COLOR["glass"], foreground=COLOR["fg"], font=self.ui_font,
            bordercolor=COLOR["control_border"], focuscolor=COLOR["accent"],
        )
        style.configure("TFrame", background=COLOR["glass"])
        style.configure("TLabel", background=COLOR["glass"], foreground=COLOR["fg"], font=self.ui_font)
        style.configure(
            "Hint.TLabel", background=COLOR["glass"], foreground=COLOR["secondary"], font=self.small_font
        )
        # 顶栏副标题：比 Hint 再淡一档，让标题自己立住
        style.configure(
            "Subtitle.TLabel", background=COLOR["glass"], foreground=COLOR["tertiary"],
            font=self.small_font,
        )

        # 普通按钮：白底 + 细边，悬停/按下/禁用/焦点四态齐全，按下不改变外框尺寸。
        # 垂直 padding 给到 6：9pt 中文的行高约 19px，再小文字就贴边甚至被裁。
        style.configure(
            "TButton", font=self.ui_font, padding=(12, 6), background="#FFFFFF",
            foreground=COLOR["fg"], bordercolor=COLOR["control_border"], focusthickness=2,
            focuscolor=COLOR["accent"], relief="flat", anchor="center",
        )
        style.map(
            "TButton",
            background=[("pressed", COLOR["hover"]), ("active", COLOR["hover"]), ("disabled", COLOR["glass"])],
            foreground=[("disabled", COLOR["disabled_fg"])],
            bordercolor=[("focus", COLOR["accent"]), ("active", COLOR["control_border"])],
        )
        # 主操作按钮：accent 实底 + 白字（4.7:1，达 AA）
        style.configure(
            "Accent.TButton", font=self.ui_font, padding=(14, 6), background=COLOR["accent"],
            foreground=COLOR["on_accent"], bordercolor=COLOR["accent"], focusthickness=2,
            focuscolor=COLOR["accent"], relief="flat", anchor="center",
        )
        style.map(
            "Accent.TButton",
            background=[("pressed", COLOR["accent_pressed"]), ("active", COLOR["accent_hover"]),
                        ("disabled", COLOR["disabled_fg"])],
            foreground=[("disabled", COLOR["glass"])],
        )
        style.configure(
            "TCombobox", padding=(8, 5), fieldbackground="#FFFFFF", background="#FFFFFF",
            arrowcolor=COLOR["secondary"], bordercolor=COLOR["control_border"],
            lightcolor="#FFFFFF", darkcolor="#FFFFFF",
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", "#FFFFFF")],
            background=[("readonly", "#FFFFFF"), ("active", COLOR["hover"])],
            arrowcolor=[("disabled", COLOR["disabled_fg"])],
            bordercolor=[("focus", COLOR["accent"])],
        )
        # 下拉展开的那张列表不是 ttk 控件，只能用选项库染色
        self.root.option_add("*TCombobox*Listbox.background", "#FFFFFF")
        self.root.option_add("*TCombobox*Listbox.foreground", COLOR["fg"])
        self.root.option_add("*TCombobox*Listbox.selectBackground", COLOR["select"])
        self.root.option_add("*TCombobox*Listbox.selectForeground", COLOR["fg"])
        self.root.option_add("*TCombobox*Listbox.font", self.ui_font)

        style.configure(
            "Treeview", font=self.ui_font, rowheight=27, background=COLOR["glass"],
            fieldbackground=COLOR["glass"], foreground=COLOR["fg"], bordercolor=COLOR["separator"],
            lightcolor=COLOR["glass"], darkcolor=COLOR["glass"],
        )
        style.map(
            "Treeview",
            background=[("selected", COLOR["select"])],
            foreground=[("selected", COLOR["fg"])],
        )
        # 表头：浅底 + 细线 + 次要文字色，去掉了 clam 默认的 3D 灰凸起
        style.configure(
            "Treeview.Heading", font=self.ui_font, padding=(8, 7), background=COLOR["hover"],
            foreground=COLOR["secondary"], relief="flat", borderwidth=1,
            bordercolor=COLOR["separator"], lightcolor=COLOR["separator"], darkcolor=COLOR["separator"],
        )
        style.map(
            "Treeview.Heading",
            background=[("active", COLOR["select"])],
            bordercolor=[("active", COLOR["control_border"])],
        )

        style.configure(
            "TNotebook", background=COLOR["glass"], bordercolor=COLOR["separator"],
            tabmargins=(8, 6, 8, 0), lightcolor=COLOR["separator"], darkcolor=COLOR["separator"],
        )
        style.configure(
            "TNotebook.Tab", font=self.ui_font, padding=(16, 8), background=COLOR["glass"],
            foreground=COLOR["secondary"], bordercolor=COLOR["separator"],
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#FFFFFF"), ("active", COLOR["hover"])],
            foreground=[("selected", COLOR["accent"])],
            bordercolor=[("selected", COLOR["separator"])],
        )
        # 进度条：细一点（6px），槽底色用「井」，两端圆润、无 3D 斜角
        style.configure(
            "TProgressbar", background=COLOR["accent"], troughcolor=COLOR["well"],
            bordercolor=COLOR["well"], lightcolor=COLOR["accent"], darkcolor=COLOR["accent"],
            thickness=6, relief="flat",
        )
        # 滚动条：clam 默认是厚实的 3D 灰条，这里压成细长扁平、悬停变色
        for orient in ("Vertical", "Horizontal"):
            style.configure(
                f"{orient}.TScrollbar", background="#D5DAE3", troughcolor=COLOR["well"],
                bordercolor=COLOR["well"], lightcolor="#D5DAE3", darkcolor="#D5DAE3",
                arrowcolor=COLOR["secondary"], relief="flat", arrowsize=12, width=11,
            )
            style.map(
                f"{orient}.TScrollbar",
                background=[("active", "#BFC6D2"), ("pressed", COLOR["secondary"])],
                arrowcolor=[("active", COLOR["fg"])],
            )

    def _build_background(self) -> None:
        """最底层铺满窗口的渐变画布，创建后立刻 lower()。"""
        self.bg_canvas = tk.Canvas(self.root, highlightthickness=0, bd=0, background=COLOR["well"])
        self.bg_canvas.place(x=0, y=0, relwidth=1, relheight=1)
        # Canvas.lower 是 tag_lower（改画布内元素顺序），要压窗口得直接调 Tk 的 lower
        try:
            self.bg_canvas.tk.call("lower", str(self.bg_canvas))
        except tk.TclError:
            pass

    def _build_cards(self) -> None:
        """五张卡片，全部 place 在 root 上。"""
        self.header_card = GlassCard(self.root)
        self.preview_card = GlassCard(self.root)
        self.list_card = GlassCard(self.root)
        self.detail_card = GlassCard(self.root)
        self.status_card = GlassCard(self.root)

    def _build_sashes(self) -> None:
        """两条可拖动的分隔条：左右分栏一条、右列上下分行一条。

        分隔条落在两张卡片之间的空隙里，底图取的是同一位置的渐变，所以不会在渐变上抠出
        一块突兀的色板；中间那根圆头短条是「可以拖」的提示。拖动时改的是布局比例，
        松手后比例保留。
        """
        self.v_sash = tk.Canvas(self.root, highlightthickness=0, bd=0, cursor="sb_h_double_arrow")
        self.h_sash = tk.Canvas(self.root, highlightthickness=0, bd=0, cursor="sb_v_double_arrow")
        for sash, orient in ((self.v_sash, "v"), (self.h_sash, "h")):
            sash.bind("<ButtonPress-1>", lambda event, o=orient: self._sash_press(event, o))
            sash.bind("<B1-Motion>", self._sash_motion)
            sash.bind("<ButtonRelease-1>", self._sash_release)
            sash.bind("<Enter>", lambda _event, o=orient: self._sash_hover(o, True))
            sash.bind("<Leave>", lambda _event, o=orient: self._sash_hover(o, False))

    def _bind_shortcuts(self) -> None:
        self.root.bind("<F5>", lambda _event: self._analyze_window())
        self.root.bind("<Control-o>", lambda _event: self._analyze_image())
        self.root.bind("<Control-r>", lambda _event: self.refresh_windows())

    # ------------------------------------------------------------------ 布局

    def _reflow(self, event=None) -> None:
        """窗口尺寸变化时重排；防抖 + 尺寸未变就跳过。"""
        if event is not None and getattr(event, "widget", None) is not self.root:
            return
        if self._reflow_job is not None:
            self.root.after_cancel(self._reflow_job)
        self._reflow_job = self.root.after(40, self._apply_layout)

    def _chrome_heights(self) -> tuple[int, int]:
        """按控件当前的真实需求算出顶栏 / 状态栏卡片高度。

        字体随 DPI 缩放变大后，「一行」需要的高度会变（125% 缩放下 9pt 中文约 40px 而不是
        32px），写死卡片高度就会把按钮文字裁掉，所以这里直接问控件要高度再加内边距。
        """
        pad = 2 * GlassCard.PADDING
        # 多给 2px 余量：正好相等时任何一点取整误差都会让最后一行贴边
        slack = 2
        header_h, status_h = HEADER_H, STATUS_H

        head_children = self.header_card.body.winfo_children()
        if len(head_children) >= 2:
            gap = self._pack_pady(head_children[1])
            need = head_children[0].winfo_reqheight() + gap + head_children[1].winfo_reqheight()
            if need > 1:
                header_h = need + pad + slack

        status_children = self.status_card.body.winfo_children()
        if status_children:
            need = max(child.winfo_reqheight() for child in status_children)
            if need > 1:
                status_h = need + pad + slack
        return header_h, status_h

    @staticmethod
    def _pack_pady(widget: tk.Misc) -> int:
        """取一个控件 pack 时上边的 pady（(8, 0) 这种取 8），拿不到就返回 0。"""
        try:
            pady = widget.pack_info().get("pady", 0)
        except (tk.TclError, AttributeError):
            return 0
        if isinstance(pady, (tuple, list)):
            return int(pady[0]) if pady else 0
        try:
            return int(pady)
        except (TypeError, ValueError):
            return 0

    def _apply_layout(self, force: bool = False) -> None:
        """按比例摆放五张卡片和两条分隔条。窗口没变尺寸且不强制时直接跳过。"""
        self._reflow_job = None
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        if width <= 1 or height <= 1:
            return
        resized = (
            abs(width - self._last_layout_size[0]) >= 2
            or abs(height - self._last_layout_size[1]) >= 2
        )
        if not resized and not force:
            return
        if resized:
            self._last_layout_size = (width, height)
            self._render_background(width, height)

        header_h, status_h = self._chrome_heights()
        self._chrome_h = (header_h, status_h)

        self._place_card(self.header_card, MARGIN, MARGIN, width - 2 * MARGIN, header_h)
        self._place_card(
            self.status_card, MARGIN, height - MARGIN - status_h, width - 2 * MARGIN, status_h
        )

        bleed = GlassCard.BLEED
        sash = GAP - 2 * bleed                       # 分隔条正好填满两卡画布之间的空隙
        body_y = MARGIN + header_h + GAP
        body_h = height - MARGIN - status_h - GAP - body_y

        # 左右分栏：可用宽度扣掉分隔条后按比例分给左列，并夹在最小宽度之间
        span = width - 2 * MARGIN - GAP
        left_w = max(MIN_LEFT_W, min(int(round(span * self._left_ratio)), span - MIN_RIGHT_W))
        right_x = MARGIN + left_w + GAP
        right_w = width - MARGIN - right_x

        # 右列上下分行
        avail_h = body_h - GAP
        list_h = max(MIN_LIST_H, min(int(round(avail_h * self._top_ratio)), avail_h - MIN_DETAIL_H))

        self._place_card(self.preview_card, MARGIN, body_y, left_w, body_h)
        self._place_card(self.list_card, right_x, body_y, right_w, list_h)
        self._place_card(self.detail_card, right_x, body_y + list_h + GAP, right_w, body_h - list_h - GAP)

        # 分隔条：并排放在相邻两卡画布之间的空隙里
        self._place_sash(self.v_sash, "v", MARGIN + left_w + bleed, body_y, sash, body_h)
        self._place_sash(self.h_sash, "h", right_x, body_y + list_h + bleed, right_w, sash)

    def _place_sash(self, sash: tk.Canvas, orient: str, x: int, y: int, w: int, h: int) -> None:
        sash.place(x=x, y=y, width=w, height=h)
        self._sash_geom[orient] = (x, y, w, h)
        self._paint_sash(orient)

    def _paint_sash(self, orient: str) -> None:
        """给分隔条铺上和周围一致的渐变底，再画中间的圆头短条（悬停/拖动时变色）。"""
        geom = self._sash_geom.get(orient)
        if geom is None:
            return
        x, y, w, h = geom
        sash = self.v_sash if orient == "v" else self.h_sash
        photo: ImageTk.PhotoImage | None = None
        try:
            patch = _gradient_background(
                max(1, self.root.winfo_width()), max(1, self.root.winfo_height())
            ).crop((x, y, x + w, y + h))
            photo = ImageTk.PhotoImage(patch)
        except Exception:
            photo = None  # 底图失败就退回单色，不影响拖动

        sash.delete("all")
        if photo is not None:
            sash._photo = photo  # 保住引用，别让 PhotoImage 被回收
            sash.create_image(0, 0, anchor="nw", image=photo)

        if self._drag is not None and self._drag[0] == orient:
            color = COLOR["accent"]
        elif self._sash_hot.get(orient):
            color = "#9BA2AD"
        else:
            color = "#C6CBD4"
        if orient == "v":
            mid = h // 2
            sash.create_line(w / 2, mid - 18, w / 2, mid + 18, width=4, capstyle="round", fill=color)
        else:
            mid = w // 2
            sash.create_line(mid - 18, h / 2, mid + 18, h / 2, width=4, capstyle="round", fill=color)

    def _sash_press(self, event, orient: str) -> None:
        if orient == "v":
            span = self.root.winfo_width() - 2 * MARGIN - GAP
            start = int(round(span * self._left_ratio))
        else:
            body_y = MARGIN + self._chrome_h[0] + GAP
            span = (
                self.root.winfo_height() - MARGIN - self._chrome_h[1] - GAP - body_y - GAP
            )
            start = int(round(span * self._top_ratio))
        self._drag = (orient, event.x_root, event.y_root, start, span)
        self._paint_sash(orient)

    def _sash_motion(self, event) -> None:
        if self._drag is None:
            return
        orient, x0, y0, start, span = self._drag
        if span <= 1:
            return
        if orient == "v":
            lo, hi = MIN_LEFT_W, max(MIN_LEFT_W, span - MIN_RIGHT_W)
            value = max(lo, min(start + (event.x_root - x0), hi))
            self._left_ratio = value / span
        else:
            lo, hi = MIN_LIST_H, max(MIN_LIST_H, span - MIN_DETAIL_H)
            value = max(lo, min(start + (event.y_root - y0), hi))
            self._top_ratio = value / span
        self._apply_layout(force=True)

    def _sash_release(self, _event) -> None:
        if self._drag is None:
            return
        orient = self._drag[0]
        self._drag = None
        self._paint_sash(orient)

    def _sash_hover(self, orient: str, hot: bool) -> None:
        self._sash_hot[orient] = hot
        if self._drag is None:
            self._paint_sash(orient)

    @staticmethod
    def _place_card(card: GlassCard, x: int, y: int, w: int, h: int) -> None:
        """按「玻璃本体」的矩形摆放卡片：画布四周各外扩 BLEED，留给投影和外发光。"""
        bleed = GlassCard.BLEED
        card.place(x=x - bleed, y=y - bleed, width=w + 2 * bleed, height=h + 2 * bleed)

    def _render_background(self, width: int, height: int) -> None:
        try:
            image = _gradient_background(width, height)
            self._bg_photo = ImageTk.PhotoImage(image)
        except Exception:
            return
        self.bg_canvas.delete("bg")
        self.bg_canvas.create_image(0, 0, anchor="nw", image=self._bg_photo, tags="bg")

    def _build_toolbar(self) -> None:
        body = self.header_card.body
        head = ttk.Frame(body)
        head.pack(side="top", fill="x")
        ttk.Label(head, text=APP_TITLE, font=self.title_font).pack(side="left")
        ttk.Label(
            head,
            text=f"by {AUTHOR} · 阴阳师【{YYS_ACCOUNT}】· 看屏幕算分数，不注入进程、不模拟点击",
            style="Subtitle.TLabel",
        ).pack(side="left", padx=(10, 0))
        # 一键直达仓库：按钮在右，地址标签贴着它左边，窄窗口下先挤掉标签再考虑按钮
        self.repo_button = ttk.Button(head, text="项目主页 ↗", command=self._open_repo)
        self.repo_button.pack(side="right")
        ttk.Label(head, text=REPO_URL, style="Subtitle.TLabel").pack(side="right", padx=(0, 8))

        bar = ttk.Frame(body)
        bar.pack(side="top", fill="x", pady=(8, 0))

        ttk.Label(bar, text="游戏窗口").pack(side="left")
        self.window_box = ttk.Combobox(bar, state="readonly", width=44, font=self.ui_font)
        self.window_box.pack(side="left", padx=(8, 4))

        self.refresh_button = ttk.Button(bar, text="刷新 (Ctrl+R)", command=self.refresh_windows)
        self.refresh_button.pack(side="left")
        self.analyze_button = ttk.Button(
            bar, text="开始分析 (F5)", style="Accent.TButton", command=self._analyze_window
        )
        self.analyze_button.pack(side="left", padx=(8, 4))
        self.open_button = ttk.Button(bar, text="打开截图… (Ctrl+O)", command=self._analyze_image)
        self.open_button.pack(side="left")

        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=110)
        self.progress.pack(side="left", padx=12)

    def _build_status(self) -> None:
        body = self.status_card.body

        self.status_text = ttk.Label(body, text="就绪", style="Hint.TLabel")
        self.status_text.pack(side="left")

        self.export_buttons = []
        for text, command in (
            ("保存标注图", self._export_image),
            ("导出 JSON", self._export_json),
            ("导出 CSV", self._export_csv),
        ):
            button = ttk.Button(body, text=text, command=command)
            button.pack(side="right", padx=(8, 0))
            self.export_buttons.append(button)

    def _open_repo(self) -> None:
        """一键直达项目主页（用系统默认浏览器打开）。"""
        webbrowser.open(REPO_URL)

    def _build_body(self) -> None:
        # 左列：截图预览
        left = self.preview_card.body
        ttk.Label(left, text="截图预览（点方框可选中对应御魂）", style="Hint.TLabel").pack(
            anchor="w", pady=(0, 8)
        )
        # 预览那口「井」：浅色 + 1px 内描边，HIG 的内嵌凹槽观感
        self.preview = tk.Canvas(
            left, background=COLOR["well"], highlightthickness=1,
            highlightbackground=COLOR["separator"], bd=0,
        )
        self.preview.pack(fill="both", expand=True)
        self.preview.bind("<Button-1>", self._on_preview_click)
        self.preview.bind("<Motion>", self._on_preview_motion)
        self.preview.bind("<Configure>", self._on_preview_resize)

        # 右上：御魂列表
        list_root = self.list_card.body
        ttk.Label(list_root, text="御魂列表", style="Hint.TLabel").pack(anchor="w", pady=(0, 8))
        list_frame = ttk.Frame(list_root)
        list_frame.pack(fill="both", expand=True)
        columns = ("no", "set", "slot", "main", "score", "verdict")
        headings = ("#", "套装", "号位", "主属性", "评分", "建议")
        widths = (34, 150, 80, 80, 52, 52)
        anchors = {"set": "w", "score": "e"}
        self.soul_tree = ttk.Treeview(
            list_frame, columns=columns, show="headings", height=8, selectmode="browse"
        )
        for name, head, width in zip(columns, headings, widths):
            self.soul_tree.heading(name, text=head)
            self.soul_tree.column(
                name,
                width=width,
                anchor=anchors.get(name, "center"),
                stretch=(name == "set"),
            )
        for verdict, tag in VERDICT_TAG.items():
            self.soul_tree.tag_configure(tag, background=VERDICT_ROW[verdict])
        self.soul_tree.tag_configure("none", background=GLASS_FILL)
        self.soul_tree.tag_configure("hover", background=COLOR["hover"])

        scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.soul_tree.yview)
        self.soul_tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.soul_tree.pack(side="left", fill="both", expand=True)
        self.soul_tree.bind("<<TreeviewSelect>>", self._on_soul_select)
        self._bind_row_hover(self.soul_tree)

        # 右下：属性明细 / 评分明细
        detail_root = self.detail_card.body
        notebook = ttk.Notebook(detail_root)
        notebook.pack(fill="both", expand=True)

        attr_tab = ttk.Frame(notebook)
        attr_columns = ("kind", "name", "old", "final", "enhance", "conf")
        attr_headings = ("类型", "属性名", "旧值", "最终值", "强化", "置信度")
        attr_widths = (64, 116, 96, 96, 56, 68)
        attr_anchors = {"name": "w", "old": "e", "final": "e", "enhance": "e", "conf": "e"}
        self.attr_tree = ttk.Treeview(
            attr_tab, columns=attr_columns, show="headings", height=6
        )
        for name, head, width in zip(attr_columns, attr_headings, attr_widths):
            self.attr_tree.heading(name, text=head)
            self.attr_tree.column(
                name, width=width, minwidth=48,
                anchor=attr_anchors.get(name, "center"), stretch=(name == "name"),
            )
        # 主属性单独一个底色；副属性/固有属性走浅斑马纹，多行时更好扫读
        self.attr_tree.tag_configure("main", background=COLOR["select"])
        self.attr_tree.tag_configure("row_even", background=GLASS_FILL)
        self.attr_tree.tag_configure("row_odd", background="#F1F3F9")
        self.attr_tree.tag_configure("hover", background=COLOR["hover"])
        attr_scroll = ttk.Scrollbar(attr_tab, orient="vertical", command=self.attr_tree.yview)
        self.attr_tree.configure(yscrollcommand=attr_scroll.set)
        attr_scroll.pack(side="right", fill="y")
        self.attr_tree.pack(side="left", fill="both", expand=True)
        self._bind_row_hover(self.attr_tree)
        notebook.add(attr_tab, text="属性明细")

        score_tab = ttk.Frame(notebook)
        self.score_text = tk.Text(
            score_tab, wrap="word", font=self.ui_font, background=GLASS_FILL,
            foreground=COLOR["fg"], relief="flat", padx=16, pady=12, highlightthickness=0, bd=0,
            spacing1=2, spacing3=3, selectbackground=COLOR["select"],
            selectforeground=COLOR["fg"], cursor="arrow",
        )
        # 评分明细的排版：小节标题加粗、分数行用 accent、警告用暖色，扫读时能一眼分块
        self.score_text.tag_configure(
            "section", foreground=COLOR["fg"], font=self.section_font, spacing1=12, spacing3=4
        )
        self.score_text.tag_configure("score", foreground=COLOR["accent"], font=self.section_font)
        self.score_text.tag_configure("warn", foreground="#B25000")
        self.score_text.tag_configure("note", foreground=COLOR["secondary"])
        score_scroll = ttk.Scrollbar(score_tab, orient="vertical", command=self.score_text.yview)
        self.score_text.configure(yscrollcommand=score_scroll.set, state="disabled")
        score_scroll.pack(side="right", fill="y")
        self.score_text.pack(side="left", fill="both", expand=True)
        notebook.add(score_tab, text="评分明细")

    # ------------------------------------------------------------------ 窗口列表

    def refresh_windows(self) -> None:
        found = [
            window
            for window in list_windows(min_size=200)
            if window.title
            and not window.title.startswith(APP_TITLE)
            and "TkTopLevel" not in window.class_name
        ]
        found.sort(key=lambda window: window.width * window.height, reverse=True)
        self.windows = found
        self.window_box["values"] = [
            f"{window.title}   [{window.width}x{window.height}]" for window in found
        ]
        if found and self.window_box.current() < 0:
            self.window_box.current(0)
        self._set_status(f"找到 {len(found)} 个可见窗口")

    # ------------------------------------------------------------------ 分析

    def _analyze_window(self) -> None:
        index = self.window_box.current()
        if index < 0 or not self.windows:
            messagebox.showwarning(APP_TITLE, "先选一个游戏窗口。")
            return
        target = self.windows[index]
        self._start(lambda: capture_client_area(target.hwnd), target.title)

    def _analyze_image(self) -> None:
        path = filedialog.askopenfilename(
            title="选一张截图",
            initialdir=str(OUT_DIR if OUT_DIR.exists() else ROOT),
            filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp"), ("所有文件", "*.*")],
        )
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror(APP_TITLE, f"读不出这张图片：\n{path}")
            return
        self._start(lambda: frame, Path(path).name)

    def _start(self, get_frame, title: str) -> None:
        if self.busy:
            return
        self.busy = True
        self.started_at = time.perf_counter()
        self._set_buttons_enabled(False)
        self._set_status("分析中…（首次运行要先加载 OCR 模型，会慢几秒）")
        threading.Thread(target=self._worker, args=(get_frame, title), daemon=True).start()

    def _worker(self, get_frame, title: str) -> None:
        try:
            frame = get_frame()
            analysis = analyze_frame(frame, self.rules, threshold=THRESHOLD, scale=SCALE)
        except Exception as exc:  # 抓窗口、OCR 都可能抛，统一报给用户
            self.root.after(0, self._on_failed, exc)
            return
        elapsed = time.perf_counter() - self.started_at
        self.root.after(0, self._on_done, frame, analysis, title, elapsed)

    def _on_failed(self, exc: Exception) -> None:
        self.busy = False
        self._set_buttons_enabled(True)
        # 状态栏留在界面上（不只弹窗），窗口最小化或弹窗被关掉时也还能看到原因
        self._set_status(f"分析失败：{exc}")
        hint = ""
        if not isinstance(exc, AnalysisError):
            hint = "\n\n如果是抓不到画面，确认游戏窗口没有最小化，且停在御魂强化结果页。"
        messagebox.showerror(APP_TITLE, f"分析失败：{exc}{hint}")

    def _on_done(self, frame: np.ndarray, analysis: PanelAnalysis, title: str, elapsed: float) -> None:
        self.frame = frame
        self.analysis = analysis
        self.source_title = title
        self.elapsed = elapsed
        self.busy = False
        self._set_buttons_enabled(True)

        self._render_preview()
        self._fill_soul_list(analysis)
        if analysis.souls:
            self._select_soul(0)
        else:
            self._clear_detail()

        mode = MODE_LABEL.get(analysis.mode, "单个御魂")
        self._set_status(
            f"{title}    {frame.shape[1]}x{frame.shape[0]}    {mode}    "
            f"{len(analysis.souls)} 件    耗时 {elapsed:.1f}s"
        )

    def _set_buttons_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in (self.refresh_button, self.analyze_button, self.open_button, *self.export_buttons):
            widget.configure(state=state)
        if enabled:
            self.progress.stop()
        else:
            self.progress.start(12)

    def _set_status(self, text: str) -> None:
        self.status_text.configure(text=text)

    # ------------------------------------------------------------------ 预览图

    def _on_preview_resize(self, event) -> None:
        if abs(event.width - self._last_canvas_size[0]) < 4 and abs(
            event.height - self._last_canvas_size[1]
        ) < 4:
            return
        self._last_canvas_size = (event.width, event.height)
        self._render_preview()

    def _render_preview(self) -> None:
        canvas = self.preview
        canvas.delete("all")
        self.photo = None

        width, height = canvas.winfo_width(), canvas.winfo_height()
        if self.frame is None:
            # 空态：淡淡的虚线占位框 + 居中的两级提示文字，比原来左上角一行小字更像回事
            if width > 60 and height > 60:
                pad = 18
                canvas.create_rectangle(
                    pad, pad, width - pad, height - pad, outline=COLOR["separator"],
                    dash=(6, 5), width=1,
                )
                cx, cy = width // 2, height // 2
            else:
                cx, cy = 24, 60
            wrap = max(160, min(width - 80, 460))
            canvas.create_text(
                cx, cy - 16, anchor="center", justify="center", fill=COLOR["fg"],
                font=self.empty_font, width=wrap, text="还没有截图",
            )
            canvas.create_text(
                cx, cy + 16, anchor="center", justify="center", fill=COLOR["secondary"],
                font=self.small_font, width=wrap,
                text="选游戏窗口点「开始分析」，或用「打开截图…」读一张图。",
            )
            return
        if width <= 1 or height <= 1:
            return

        photo, scale = _to_photo(self.frame, width, height)
        self.photo = photo
        self.image_scale = scale
        canvas.create_image(0, 0, anchor="nw", image=photo)
        self._draw_boxes()

    def _draw_boxes(self) -> None:
        """画每件御魂的方框；选中的那个加粗并配一个实底标签，和右侧列表一一对应。"""
        canvas = self.preview
        canvas.delete("box")
        self.card_boxes.clear()
        if self.analysis is None:
            return
        scale = self.image_scale
        for soul in self.analysis.souls:
            box = soul.box
            if box is None:
                continue
            x0, y0 = box.x * scale, box.y * scale
            x1, y1 = box.x1 * scale, box.y1 * scale
            result = soul.result
            mark = VERDICT_MARK.get(result.verdict if result else "", COLOR["secondary"])
            selected = soul.index == self.selected_index
            canvas.create_rectangle(
                x0, y0, x1, y1, outline=mark, width=3 if selected else 2, tags="box"
            )
            label = f"{soul.index + 1} · {result.score:.0f} {result.verdict}" if result else f"{soul.index + 1}"
            # 先画一遍白色（偏移 (1,1) 当描边），再画彩色——压在任意游戏截图上都读得清
            canvas.create_text(
                x0 + 6 + 1, y0 + 6 + 1, anchor="nw", fill="#FFFFFF", font=self.small_font,
                text=label, tags="box",
            )
            text_id = canvas.create_text(
                x0 + 6, y0 + 6, anchor="nw", fill=mark, font=self.small_font,
                text=label, tags="box",
            )
            if selected:
                bounds = canvas.bbox(text_id)
                if bounds:
                    chip = canvas.create_rectangle(
                        bounds[0] - 4, bounds[1] - 2, bounds[2] + 4, bounds[3] + 2,
                        fill="#FFFFFF", outline=mark, tags="box",
                    )
                    canvas.tag_lower(chip, text_id)
            self.card_boxes.append((x0, y0, x1, y1, soul.index))

    def _on_preview_motion(self, event) -> None:
        over = any(x0 <= event.x <= x1 and y0 <= event.y <= y1 for x0, y0, x1, y1, _ in self.card_boxes)
        cursor = "hand2" if over else ""
        if self.preview["cursor"] != cursor:
            self.preview.configure(cursor=cursor)

    def _on_preview_click(self, event) -> None:
        for x0, y0, x1, y1, index in self.card_boxes:
            if x0 <= event.x <= x1 and y0 <= event.y <= y1:
                self._select_soul(index)
                return

    # ------------------------------------------------------------------ 明细

    def _bind_row_hover(self, tree: ttk.Treeview) -> None:
        tree.bind("<Motion>", self._on_tree_motion)
        tree.bind("<Leave>", self._on_tree_leave)

    def _on_tree_motion(self, event) -> None:
        """鼠标划过表格行时给该行加一层浅色底。

        ttk 里同一个选项被多个 tag 指定时，排在前面的 tag 生效，所以只要把 hover 插到
        标签列表最前面，就能压住原有的建议色底，同时其它选项不受影响；离开时摘掉还原。
        """
        tree = event.widget
        iid = tree.identify_row(event.y)
        if iid == getattr(tree, "_hover_iid", None):
            return
        self._clear_tree_hover(tree)
        if iid and tree.exists(iid):
            base = [tag for tag in tree.item(iid, "tags") if tag != "hover"]
            tree.item(iid, tags=("hover", *base))
            tree._hover_iid = iid

    def _on_tree_leave(self, event) -> None:
        self._clear_tree_hover(event.widget)

    @staticmethod
    def _clear_tree_hover(tree: ttk.Treeview) -> None:
        iid = getattr(tree, "_hover_iid", None)
        tree._hover_iid = None
        if iid and tree.exists(iid):
            tree.item(iid, tags=[tag for tag in tree.item(iid, "tags") if tag != "hover"])

    def _fill_soul_list(self, analysis: PanelAnalysis) -> None:
        tree = self.soul_tree
        tree.delete(*tree.get_children())
        for soul in analysis.souls:
            result = soul.result
            set_name = soul.set_name or "未识别"
            if result is not None and not result.set_verified:
                set_name += "（乐观）"
            if result is not None and result.slot:
                slot = f"{result.slot}号位" + ("(推断)" if result.slot_inferred else "")
            else:
                slot = "—"
            tree.insert(
                "",
                "end",
                iid=str(soul.index),
                values=(
                    soul.index + 1,
                    set_name,
                    slot,
                    ATTR_ID_TO_NAME.get(soul.main_attr or "", "—"),
                    f"{result.score:.0f}" if result else "—",
                    result.verdict if result else "—",
                ),
                tags=(VERDICT_TAG.get(result.verdict if result else "", "none"),),
            )

    def _select_soul(self, index: int) -> None:
        iid = str(index)
        if self.soul_tree.exists(iid):
            self.soul_tree.selection_set(iid)
            self.soul_tree.focus(iid)
            self.soul_tree.see(iid)
        self._show_detail(index)

    def _on_soul_select(self, _event) -> None:
        selection = self.soul_tree.selection()
        if selection:
            self._show_detail(int(selection[0]))

    def _clear_detail(self) -> None:
        self.selected_index = -1
        self._draw_boxes()
        self.attr_tree.delete(*self.attr_tree.get_children())
        self.score_text.configure(state="normal")
        self.score_text.delete("1.0", "end")
        self.score_text.configure(state="disabled")

    def _show_detail(self, index: int) -> None:
        if self.analysis is None or not (0 <= index < len(self.analysis.souls)):
            self._clear_detail()
            return
        soul = self.analysis.souls[index]
        if index != self.selected_index:
            self.selected_index = index
            self._draw_boxes()

        self.attr_tree.delete(*self.attr_tree.get_children())
        stripe = 0
        for row_index, item in soul.indexed:
            if soul.badges:
                mark = str(soul.badges.get(row_index, 0))
            else:
                mark = "是" if item["upgraded"] else "否"
            if item["kind"] == "main":
                tag = "main"
            else:
                tag = "row_even" if stripe % 2 == 0 else "row_odd"
                stripe += 1
            self.attr_tree.insert(
                "",
                "end",
                values=(
                    KIND_LABEL.get(item["kind"], item["kind"]),
                    item["name"],
                    format_value(item["old"], item["unit"]),
                    format_value(item["final"], item["unit"]),
                    mark,
                    f"{item['score']:.2f}",
                ),
                tags=(tag,),
            )

        self._render_score(soul)

    def _render_score(self, soul: SoulReport) -> None:
        """把评分明细按行贴进 Text，并按行内容挂上排版标签（小节 / 分数 / 警告 / 提示）。"""
        text = self.score_text
        text.configure(state="normal")
        text.delete("1.0", "end")
        for line in self._detail_text(soul).split("\n"):
            stripped = line.strip()
            if line.startswith("评分："):
                tag = ("score",)
            elif stripped.startswith(("计分明细", "备选方案", "没解析出来的行")):
                tag = ("section",)
            elif stripped.startswith("[警告]"):
                tag = ("warn",)
            elif line.startswith("提示："):
                tag = ("note",)
            else:
                tag = ()
            text.insert("end", line + "\n", tag)
        text.configure(state="disabled")

    def _detail_text(self, soul: SoulReport) -> str:
        lines: list[str] = []
        set_match = soul.set_match
        if set_match is None:
            lines.append("套装：未识别出")
        else:
            lines.append(
                f"套装：{set_match.name}（匹配分数 {set_match.score:.3f}，领先 {set_match.margin:.3f}）"
            )

        result = soul.result
        if result is None:
            lines.append("评分：没有命中的方案")
        else:
            slot = f"{result.slot}号位" if result.slot else "号位未知"
            if result.slot_inferred:
                slot += "（推断）"
            lines.append("")
            lines.append(
                f"评分：{result.score:.0f}    方案：{result.rule_name}    "
                f"号位：{slot}    建议：{result.verdict}"
            )
            if not result.set_verified:
                lines.append("提示：套装没识别出来，这个分数是把所有套装方案都拿来试的最高值（乐观上界）。")
            lines.append("")
            lines.append("计分明细（核心属性 10 分/次，一般属性 5 分/次，主属性不计分）：")
            if result.contributions:
                for contribution in result.contributions:
                    kind = "核心" if contribution.kind == "core" else "一般"
                    lines.append(
                        f"  {kind}  {contribution.name}  强化 {contribution.enhancement_count} 次"
                        f"   {contribution.weight:.0f} × {contribution.enhancement_count} + 基础"
                        f"  →  {contribution.score:.0f} 分"
                    )
            else:
                lines.append("  （没有任何属性落在这个方案的核心/一般属性里）")
            if len(result.candidates) > 1:
                lines.append("")
                lines.append("备选方案：")
                for name, score, verdict in result.candidates[1:]:
                    lines.append(f"  {name}    {score:.0f} 分    {verdict}")

        for warning in soul.warnings:
            lines.append("")
            lines.append(f"[警告] {warning}")
        bad_lines = unparsed(soul.lines, kinds=soul.kinds)
        if bad_lines:
            lines.append("")
            lines.append("没解析出来的行：")
            for line in bad_lines:
                lines.append(f"  {line.text!r}（置信度 {line.score:.2f}）")
        return "\n".join(lines)

    # ------------------------------------------------------------------ 导出

    def _require_result(self) -> bool:
        if self.analysis is None or self.frame is None:
            messagebox.showinfo(APP_TITLE, "还没有分析结果。")
            return False
        return True

    def _ask_path(self, suffix: str, default: str, filetypes) -> str | None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            initialdir=str(OUT_DIR), initialfile=default, defaultextension=suffix,
            filetypes=filetypes,
        )
        return path or None

    def _export_json(self) -> None:
        if not self._require_result():
            return
        path = self._ask_path(".json", "panel_read.json", [("JSON", "*.json")])
        if path is None:
            return
        payload = self.analysis.to_dict(self.source_title, time.strftime("%Y-%m-%d %H:%M:%S"))
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._set_status(f"JSON 已导出：{path}")

    def _export_csv(self) -> None:
        if not self._require_result():
            return
        path = self._ask_path(".csv", "panel_read.csv", [("CSV", "*.csv")])
        if path is None:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "soul", "kind", "name", "old", "final", "unit", "upgraded",
                "ocr_score", "enhance_count", "soul_score", "rule", "verdict",
            ])
            for soul in self.analysis.souls:
                score = soul.result.to_dict() if soul.result else {}
                for row_index, item in soul.indexed:
                    writer.writerow([
                        soul.index + 1, item["kind"], item["name"], item["old"],
                        item["final"], item["unit"], int(item["upgraded"]),
                        f"{item['score']:.3f}", soul.badges.get(row_index, 0),
                        score.get("score"), score.get("rule_name"), score.get("verdict"),
                    ])
        self._set_status(f"CSV 已导出：{path}")

    def _annotated_image(self) -> np.ndarray:
        """在截图上画每件御魂的方框和分数，颜色对应 留/观察/弃。"""
        lines = [line for soul in self.analysis.souls for line in soul.lines]
        canvas = annotate(self.frame, lines)
        for soul in self.analysis.souls:
            box = soul.box
            if box is None:
                continue
            verdict = soul.result.verdict if soul.result else ""
            color = VERDICT_BGR.get(verdict, (200, 120, 60))
            cv2.rectangle(canvas, (box.x, box.y), (box.x1, box.y1), color, 2)
            score = f"{soul.result.score:.0f}" if soul.result else "-"
            # cv2.putText 画不了中文，用序号 + 分数 + 颜色区分
            cv2.putText(
                canvas, f"#{soul.index + 1} {score}", (box.x + 2, max(box.y - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )
        return canvas

    def _export_image(self) -> None:
        if not self._require_result():
            return
        path = self._ask_path(".png", "annotated.png", [("PNG", "*.png")])
        if path is None:
            return
        cv2.imwrite(path, self._annotated_image())
        self._set_status(f"标注图已保存：{path}")


def _apply_font_scaling(root: tk.Tk) -> None:
    """进程开了 DPI 感知后 Tk 不会自己放大字体，按系统 DPI 补上，免得字太小。"""
    try:
        dpi = ctypes.windll.user32.GetDpiForSystem()
    except Exception:
        return
    if dpi:
        root.tk.call("tk", "scaling", dpi / 72.0)


def main() -> int:
    enable_dpi_awareness()
    root = tk.Tk()
    _apply_font_scaling(root)
    SoulScorerApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
