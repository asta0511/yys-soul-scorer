"""Windows 窗口枚举与截图。

提供 4 种捕获后端，用于逐个试探游戏窗口（DX 渲染）能否正常截图：
    printwindow_full  - PrintWindow + PW_RENDERFULLCONTENT，可截后台 DX 窗口
    printwindow_plain - PrintWindow 默认标志，部分老引擎只认这个
    bitblt            - 从窗口 DC 直接 BitBlt，GDI 渲染窗口有效
    client_area       - 只截客户区（去掉标题栏边框），同样走 PrintWindow
    screen_region     - 截取窗口在屏幕上的矩形区域，要求窗口前台可见

统一输出 BGR 顺序的 uint8 ndarray，形状 (h, w, 3)，可直接交给 OpenCV。

注意：所有 Win32 调用都显式声明 argtypes，否则 ctypes 会把 64 位句柄按
32 位 int 传递，抛出 OverflowError。
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
from dataclasses import dataclass

import numpy as np

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
dwmapi = ctypes.WinDLL("dwmapi", use_last_error=True)

PW_CLIENTONLY = 0x00000001
PW_RENDERFULLCONTENT = 0x00000002
SRCCOPY = 0x00CC0020
BI_RGB = 0
DIB_RGB_COLORS = 0
DWMWA_EXTENDED_FRAME_BOUNDS = 9
SW_RESTORE = 9


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wt.DWORD),
        ("biWidth", ctypes.c_long),
        ("biHeight", ctypes.c_long),
        ("biPlanes", wt.WORD),
        ("biBitCount", wt.WORD),
        ("biCompression", wt.DWORD),
        ("biSizeImage", wt.DWORD),
        ("biXPelsPerMeter", ctypes.c_long),
        ("biYPelsPerMeter", ctypes.c_long),
        ("biClrUsed", wt.DWORD),
        ("biClrImportant", wt.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wt.DWORD * 3)]


WNDENUMPROC = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)

user32.EnumWindows.argtypes = [WNDENUMPROC, wt.LPARAM]
user32.EnumWindows.restype = wt.BOOL
user32.IsWindowVisible.argtypes = [wt.HWND]
user32.IsWindowVisible.restype = wt.BOOL
user32.GetWindowTextW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
user32.GetWindowTextW.restype = ctypes.c_int
user32.GetWindowTextLengthW.argtypes = [wt.HWND]
user32.GetWindowTextLengthW.restype = ctypes.c_int
user32.GetClassNameW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
user32.GetClassNameW.restype = ctypes.c_int
user32.GetWindowRect.argtypes = [wt.HWND, ctypes.POINTER(wt.RECT)]
user32.GetWindowRect.restype = wt.BOOL
user32.GetClientRect.argtypes = [wt.HWND, ctypes.POINTER(wt.RECT)]
user32.GetClientRect.restype = wt.BOOL
user32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
user32.GetWindowThreadProcessId.restype = wt.DWORD
user32.GetWindowDC.argtypes = [wt.HWND]
user32.GetWindowDC.restype = wt.HDC
user32.GetDC.argtypes = [wt.HWND]
user32.GetDC.restype = wt.HDC
user32.ReleaseDC.argtypes = [wt.HWND, wt.HDC]
user32.ReleaseDC.restype = ctypes.c_int
user32.PrintWindow.argtypes = [wt.HWND, wt.HDC, wt.UINT]
user32.PrintWindow.restype = wt.BOOL
user32.ShowWindow.argtypes = [wt.HWND, ctypes.c_int]
user32.ShowWindow.restype = wt.BOOL
user32.SetForegroundWindow.argtypes = [wt.HWND]
user32.SetForegroundWindow.restype = wt.BOOL

gdi32.CreateCompatibleDC.argtypes = [wt.HDC]
gdi32.CreateCompatibleDC.restype = wt.HDC
gdi32.CreateCompatibleBitmap.argtypes = [wt.HDC, ctypes.c_int, ctypes.c_int]
gdi32.CreateCompatibleBitmap.restype = wt.HANDLE
gdi32.SelectObject.argtypes = [wt.HDC, wt.HANDLE]
gdi32.SelectObject.restype = wt.HANDLE
gdi32.DeleteObject.argtypes = [wt.HANDLE]
gdi32.DeleteObject.restype = wt.BOOL
gdi32.DeleteDC.argtypes = [wt.HDC]
gdi32.DeleteDC.restype = wt.BOOL
gdi32.BitBlt.argtypes = [
    wt.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    wt.HDC, ctypes.c_int, ctypes.c_int, wt.DWORD,
]
gdi32.BitBlt.restype = wt.BOOL
gdi32.GetDIBits.argtypes = [
    wt.HDC, wt.HANDLE, wt.UINT, wt.UINT, ctypes.c_void_p,
    ctypes.POINTER(BITMAPINFO), wt.UINT,
]
gdi32.GetDIBits.restype = ctypes.c_int

dwmapi.DwmGetWindowAttribute.argtypes = [wt.HWND, wt.DWORD, ctypes.c_void_p, wt.DWORD]
dwmapi.DwmGetWindowAttribute.restype = ctypes.c_long


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    class_name: str
    left: int
    top: int
    right: int
    bottom: int
    pid: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom

    def __str__(self) -> str:
        return (
            f"hwnd=0x{self.hwnd:08X} pid={self.pid} "
            f"pos=({self.left},{self.top}) size={self.width}x{self.height} "
            f"class={self.class_name!r} title={self.title!r}"
        )


def enable_dpi_awareness() -> str:
    """让本进程按物理像素工作，否则窗口坐标和截屏坐标会错位。"""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_AWARE
        return "per-monitor"
    except Exception:
        pass
    try:
        user32.SetProcessDPIAware()
        return "system"
    except Exception:
        return "none"


def _extended_frame_bounds(hwnd: int) -> tuple[int, int, int, int] | None:
    """Win10 起 GetWindowRect 含不可见边框，DWM 返回的才是真实可见区域。"""
    rect = wt.RECT()
    res = dwmapi.DwmGetWindowAttribute(
        hwnd, DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(rect), ctypes.sizeof(rect)
    )
    if res != 0:
        return None
    return rect.left, rect.top, rect.right, rect.bottom


def _pid_of(hwnd: int) -> int:
    pid = wt.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value


def list_windows(visible_only: bool = True, min_size: int = 0) -> list[WindowInfo]:
    """枚举顶层窗口。min_size 会过滤掉宽或高小于该值的窗口。"""
    found: list[WindowInfo] = []

    def _callback(hwnd, _lparam):
        if visible_only and not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True

        title_buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, title_buf, length + 1)

        class_buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, class_buf, 256)

        bounds = _extended_frame_bounds(hwnd)
        if bounds is None:
            rect = wt.RECT()
            if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                return True
            bounds = (rect.left, rect.top, rect.right, rect.bottom)

        left, top, right, bottom = bounds
        if right - left < min_size or bottom - top < min_size:
            return True

        found.append(
            WindowInfo(
                hwnd=hwnd,
                title=title_buf.value,
                class_name=class_buf.value,
                left=left,
                top=top,
                right=right,
                bottom=bottom,
                pid=_pid_of(hwnd),
            )
        )
        return True

    user32.EnumWindows(WNDENUMPROC(_callback), 0)
    return found


def find_windows(keyword: str, visible_only: bool = True, min_size: int = 0) -> list[WindowInfo]:
    """按标题或窗口类名做不区分大小写的子串匹配。"""
    needle = keyword.lower()
    return [
        w
        for w in list_windows(visible_only=visible_only, min_size=min_size)
        if needle in w.title.lower() or needle in w.class_name.lower()
    ]


def bring_to_front(hwnd: int, settle: float = 0.6) -> bool:
    import time

    user32.ShowWindow(hwnd, SW_RESTORE)
    ok = bool(user32.SetForegroundWindow(hwnd))
    time.sleep(settle)
    return ok


def _bitmap_to_bgr(hdc: int, hbmp: int, width: int, height: int) -> np.ndarray:
    header = BITMAPINFOHEADER()
    header.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    header.biWidth = width
    header.biHeight = -height  # 负数表示自上而下，省一次翻转
    header.biPlanes = 1
    header.biBitCount = 32
    header.biCompression = BI_RGB

    info = BITMAPINFO()
    info.bmiHeader = header

    buffer = ctypes.create_string_buffer(width * height * 4)
    got = gdi32.GetDIBits(hdc, hbmp, 0, height, buffer, ctypes.byref(info), DIB_RGB_COLORS)
    if got == 0:
        raise RuntimeError("GetDIBits 失败")

    raw = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 4)
    return raw[:, :, :3][:, :, ::-1].copy()


def _grab_window(hwnd: int, draw) -> np.ndarray:
    """申请内存 DC，交给 draw(hdc_window, hdc_mem, w, h) 绘制，再取回像素。"""
    rect = wt.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        raise ctypes.WinError(ctypes.get_last_error())
    width, height = rect.right - rect.left, rect.bottom - rect.top
    if width <= 0 or height <= 0:
        raise RuntimeError(f"窗口尺寸非法: {width}x{height}")

    hdc_window = user32.GetWindowDC(hwnd)
    if not hdc_window:
        raise ctypes.WinError(ctypes.get_last_error())

    hdc_mem = gdi32.CreateCompatibleDC(hdc_window)
    hbmp = gdi32.CreateCompatibleBitmap(hdc_window, width, height)
    if not hdc_mem or not hbmp:
        user32.ReleaseDC(hwnd, hdc_window)
        raise ctypes.WinError(ctypes.get_last_error())

    previous = gdi32.SelectObject(hdc_mem, hbmp)
    try:
        draw(hdc_window, hdc_mem, width, height)
        return _bitmap_to_bgr(hdc_mem, hbmp, width, height)
    finally:
        gdi32.SelectObject(hdc_mem, previous)
        gdi32.DeleteObject(hbmp)
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(hwnd, hdc_window)


def capture_printwindow(hwnd: int, full_content: bool = True) -> np.ndarray:
    flags = PW_RENDERFULLCONTENT if full_content else 0

    def _draw(_src, dst, _w, _h):
        if not user32.PrintWindow(hwnd, dst, flags):
            raise RuntimeError(f"PrintWindow 失败 (flags={flags})")

    return _grab_window(hwnd, _draw)


def capture_bitblt(hwnd: int) -> np.ndarray:
    def _draw(src, dst, width, height):
        if not gdi32.BitBlt(dst, 0, 0, width, height, src, 0, 0, SRCCOPY):
            raise RuntimeError("BitBlt 失败")

    return _grab_window(hwnd, _draw)


def capture_client_area(hwnd: int) -> np.ndarray:
    """只截客户区（去掉标题栏和边框）。"""
    rect = wt.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        raise ctypes.WinError(ctypes.get_last_error())
    width, height = rect.right, rect.bottom
    if width <= 0 or height <= 0:
        raise RuntimeError(f"客户区尺寸非法: {width}x{height}")

    hdc_window = user32.GetDC(hwnd)
    hdc_mem = gdi32.CreateCompatibleDC(hdc_window)
    hbmp = gdi32.CreateCompatibleBitmap(hdc_window, width, height)
    previous = gdi32.SelectObject(hdc_mem, hbmp)
    try:
        if not user32.PrintWindow(hwnd, hdc_mem, PW_CLIENTONLY | PW_RENDERFULLCONTENT):
            raise RuntimeError("PrintWindow(客户区) 失败")
        return _bitmap_to_bgr(hdc_mem, hbmp, width, height)
    finally:
        gdi32.SelectObject(hdc_mem, previous)
        gdi32.DeleteObject(hbmp)
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(hwnd, hdc_window)


def capture_screen_region(bbox: tuple[int, int, int, int]) -> np.ndarray:
    import mss

    left, top, right, bottom = bbox
    region = {"left": left, "top": top, "width": right - left, "height": bottom - top}
    with mss.mss() as sct:
        shot = sct.grab(region)
    return np.array(shot)[:, :, :3].copy()


BACKENDS = {
    "printwindow_full": lambda w: capture_printwindow(w.hwnd, full_content=True),
    "printwindow_plain": lambda w: capture_printwindow(w.hwnd, full_content=False),
    "bitblt": lambda w: capture_bitblt(w.hwnd),
    "client_area": lambda w: capture_client_area(w.hwnd),
    "screen_region": lambda w: capture_screen_region(w.bbox),
}
