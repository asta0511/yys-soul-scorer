"""御魂评分器的图形界面。

从截图到评分一条命令跑完：选窗口（或打开一张截图）-> 点「开始分析」-> 左边看截图、
右边看每件御魂的分数和建议，点某一件在下面看属性明细和评分依据，需要时导出
JSON / CSV / 标注图。

识别逻辑全在 src/pipeline.py，这里只做「取图 + 展示 + 导出」。分析放在子线程里跑
（一屏六张卡要 5 秒上下），跑完用 after() 回主线程刷界面——Tk 的控件只能在主线程碰。

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
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.ocr import annotate
from src.parse import format_value, unparsed
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
OUT_DIR = ROOT / "output" / "panel_read"

# 与命令行保持一致：不改这两个参数，界面和 CLI 的结果就能对上
THRESHOLD = 100
SCALE = 2

KIND_LABEL = {"main": "主属性", "sub": "副属性", "innate": "固有"}
VERDICT_TAG = {"留": "keep", "观察": "observe", "弃": "discard"}
VERDICT_HEX = {"留": "#1c7c33", "观察": "#b26a00", "弃": "#777777"}
VERDICT_BGR = {"留": (60, 170, 60), "观察": (0, 150, 230), "弃": (150, 150, 150)}

MODE_LABEL = {
    MODE_BATCH: "批量强化",
    MODE_SINGLE_CARD: "单个御魂",
    MODE_SOUL_DETAIL: "御魂详情",
}


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
        self._last_canvas_size = (0, 0)

        self.ui_font = ("Microsoft YaHei UI", 9)
        self.small_font = ("Microsoft YaHei UI", 8)

        root.title(APP_TITLE)
        root.geometry("1280x820")
        root.minsize(960, 620)
        self._init_style()
        self._build_toolbar()
        self._build_status()
        self._build_body()
        self.refresh_windows()

    # ------------------------------------------------------------------ 界面搭建

    def _init_style(self) -> None:
        style = ttk.Style()
        style.configure("Treeview", font=self.ui_font, rowheight=22)
        style.configure("Treeview.Heading", font=self.ui_font)
        style.configure("TNotebook.Tab", font=self.ui_font, padding=(10, 4))
        style.configure("TButton", font=self.ui_font)

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self.root, padding=(8, 6, 8, 2))
        bar.pack(side="top", fill="x")

        ttk.Label(bar, text="游戏窗口", font=self.ui_font).pack(side="left")
        self.window_box = ttk.Combobox(bar, state="readonly", width=44, font=self.ui_font)
        self.window_box.pack(side="left", padx=(6, 4))

        self.refresh_button = ttk.Button(bar, text="刷新", width=6, command=self.refresh_windows)
        self.refresh_button.pack(side="left")
        self.analyze_button = ttk.Button(bar, text="开始分析", command=self._analyze_window)
        self.analyze_button.pack(side="left", padx=(10, 4))
        self.open_button = ttk.Button(bar, text="打开截图…", command=self._analyze_image)
        self.open_button.pack(side="left")

        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=110)
        self.progress.pack(side="left", padx=12)

    def _build_status(self) -> None:
        bar = ttk.Frame(self.root, padding=(8, 4, 8, 8))
        bar.pack(side="bottom", fill="x")

        self.status_text = ttk.Label(bar, text="就绪", font=self.ui_font, foreground="#444")
        self.status_text.pack(side="left")

        self.export_buttons = []
        for text, command in (
            ("保存标注图", self._export_image),
            ("导出 JSON", self._export_json),
            ("导出 CSV", self._export_csv),
        ):
            button = ttk.Button(bar, text=text, width=11, command=command)
            button.pack(side="right", padx=(4, 0))
            self.export_buttons.append(button)

    def _build_body(self) -> None:
        outer = ttk.PanedWindow(self.root, orient="horizontal")
        outer.pack(side="top", fill="both", expand=True, padx=8, pady=4)

        left = ttk.Frame(outer)
        ttk.Label(
            left, text="截图预览（点方框可选中对应御魂）", font=self.small_font, foreground="#777"
        ).pack(anchor="w", pady=(0, 2))
        self.preview = tk.Canvas(left, background="#1e1e1e", highlightthickness=0)
        self.preview.pack(fill="both", expand=True)
        self.preview.bind("<Button-1>", self._on_preview_click)
        self.preview.bind("<Configure>", self._on_preview_resize)
        outer.add(left, weight=5)

        right = ttk.PanedWindow(outer, orient="vertical")

        list_frame = ttk.Frame(right)
        ttk.Label(list_frame, text="御魂列表", font=self.small_font, foreground="#777").pack(
            anchor="w", pady=(0, 2)
        )
        columns = ("no", "set", "slot", "main", "score", "verdict")
        headings = ("#", "套装", "号位", "主属性", "评分", "建议")
        widths = (34, 150, 80, 80, 52, 52)
        self.soul_tree = ttk.Treeview(
            list_frame, columns=columns, show="headings", height=8, selectmode="browse"
        )
        for name, head, width in zip(columns, headings, widths):
            self.soul_tree.heading(name, text=head)
            self.soul_tree.column(
                name,
                width=width,
                anchor="w" if name == "set" else "center",
                stretch=(name == "set"),
            )
        for verdict, tag in VERDICT_TAG.items():
            self.soul_tree.tag_configure(tag, background={
                "留": "#e4f6e7", "观察": "#fdf3dc", "弃": "#f2f2f2",
            }[verdict])
        self.soul_tree.tag_configure("none", background="#ffffff")

        scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.soul_tree.yview)
        self.soul_tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.soul_tree.pack(side="left", fill="both", expand=True)
        self.soul_tree.bind("<<TreeviewSelect>>", self._on_soul_select)
        right.add(list_frame, weight=3)

        notebook = ttk.Notebook(right)

        attr_tab = ttk.Frame(notebook)
        attr_columns = ("kind", "name", "old", "final", "enhance", "conf")
        attr_headings = ("类型", "属性名", "旧值", "最终值", "强化", "置信度")
        attr_widths = (60, 90, 90, 90, 50, 60)
        self.attr_tree = ttk.Treeview(
            attr_tab, columns=attr_columns, show="headings", height=6
        )
        for name, head, width in zip(attr_columns, attr_headings, attr_widths):
            self.attr_tree.heading(name, text=head)
            self.attr_tree.column(
                name, width=width, anchor="w" if name == "name" else "center", stretch=False
            )
        attr_scroll = ttk.Scrollbar(attr_tab, orient="vertical", command=self.attr_tree.yview)
        self.attr_tree.configure(yscrollcommand=attr_scroll.set)
        attr_scroll.pack(side="right", fill="y")
        self.attr_tree.pack(side="left", fill="both", expand=True)
        notebook.add(attr_tab, text="属性明细")

        score_tab = ttk.Frame(notebook)
        self.score_text = tk.Text(
            score_tab, wrap="word", font=self.ui_font, background="#fbfbfb",
            relief="flat", padx=10, pady=8,
        )
        score_scroll = ttk.Scrollbar(score_tab, orient="vertical", command=self.score_text.yview)
        self.score_text.configure(yscrollcommand=score_scroll.set, state="disabled")
        score_scroll.pack(side="right", fill="y")
        self.score_text.pack(side="left", fill="both", expand=True)
        notebook.add(score_tab, text="评分明细")

        right.add(notebook, weight=4)
        outer.add(right, weight=6)

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
        self.root.after(0, self._on_done, frame, analysis, title)

    def _on_failed(self, exc: Exception) -> None:
        self.busy = False
        self._set_buttons_enabled(True)
        self._set_status("分析失败")
        hint = ""
        if not isinstance(exc, AnalysisError):
            hint = "\n\n如果是抓不到画面，确认游戏窗口没有最小化，且停在御魂强化结果页。"
        messagebox.showerror(APP_TITLE, f"分析失败：{exc}{hint}")

    def _on_done(self, frame: np.ndarray, analysis: PanelAnalysis, title: str) -> None:
        self.frame = frame
        self.analysis = analysis
        self.source_title = title
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
            f"{title}    {frame.shape[1]}x{frame.shape[0]}    {mode}    {len(analysis.souls)} 件"
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
        self.card_boxes.clear()

        width, height = canvas.winfo_width(), canvas.winfo_height()
        if self.frame is None:
            canvas.create_text(
                24, 24, anchor="nw", fill="#9a9a9a", font=self.ui_font,
                text="还没有截图。\n\n选一个游戏窗口后点「开始分析」，或用「打开截图…」读一张已有的图。",
            )
            return
        if width <= 1 or height <= 1:
            return

        photo, scale = _to_photo(self.frame, width, height)
        self.photo = photo
        self.image_scale = scale
        canvas.create_image(0, 0, anchor="nw", image=photo)

        if self.analysis is None:
            return
        for soul in self.analysis.souls:
            box = soul.box
            if box is None:
                continue
            x0, y0 = box.x * scale, box.y * scale
            x1, y1 = box.x1 * scale, box.y1 * scale
            verdict = soul.result.verdict if soul.result else ""
            score = f"{soul.result.score:.0f}" if soul.result else "—"
            color = VERDICT_HEX.get(verdict, "#3a7bd5")
            canvas.create_rectangle(x0, y0, x1, y1, outline=color, width=2)
            canvas.create_text(
                x0 + 2, y0 + 2, anchor="nw", fill=color, font=self.small_font,
                text=f"{soul.index + 1} · {score} {verdict}",
            )
            self.card_boxes.append((x0, y0, x1, y1, soul.index))

    def _on_preview_click(self, event) -> None:
        for x0, y0, x1, y1, index in self.card_boxes:
            if x0 <= event.x <= x1 and y0 <= event.y <= y1:
                self._select_soul(index)
                return

    # ------------------------------------------------------------------ 明细

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
        self.attr_tree.delete(*self.attr_tree.get_children())
        self.score_text.configure(state="normal")
        self.score_text.delete("1.0", "end")
        self.score_text.configure(state="disabled")

    def _show_detail(self, index: int) -> None:
        if self.analysis is None or not (0 <= index < len(self.analysis.souls)):
            self._clear_detail()
            return
        soul = self.analysis.souls[index]

        self.attr_tree.delete(*self.attr_tree.get_children())
        for row_index, item in soul.indexed:
            if soul.badges:
                mark = str(soul.badges.get(row_index, 0))
            else:
                mark = "是" if item["upgraded"] else "否"
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
            )

        self.score_text.configure(state="normal")
        self.score_text.delete("1.0", "end")
        self.score_text.insert("1.0", self._detail_text(soul))
        self.score_text.configure(state="disabled")

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
