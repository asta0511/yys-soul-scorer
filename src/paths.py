"""资源目录解析：同一份代码要能在「源码运行」和「PyInstaller 打包后」两种方式下找到文件。

只读资源（rules/、data/）打包后被 PyInstaller 解到临时目录 sys._MEIPASS，源码运行时
就是仓库根目录；可写产物（output/）不能写进那个临时目录（进程一退就没了），要写到
exe 旁边。
"""

from __future__ import annotations

import sys
from pathlib import Path


def resource_root() -> Path:
    """只读资源（rules/、data/）所在目录。"""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return Path(__file__).resolve().parents[1]


def writable_root() -> Path:
    """可写目录（output/）的根：打包后取 exe 所在目录，源码运行时取仓库根。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]
