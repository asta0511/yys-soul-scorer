@echo off
rem ============================================================
rem  御魂评分器 一键启动
rem  双击本文件即可打开图形界面（不需要手动敲命令）
rem  说明：内容用英文是为了避免 cmd 中文编码乱码，不影响使用
rem ============================================================
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\pythonw.exe" goto :no_venv
if not exist "tools\gui.py" goto :no_code

start "" ".venv\Scripts\pythonw.exe" "tools\gui.py"
exit /b 0

:no_venv
echo [ERROR] Virtual environment not found: "%~dp0.venv"
echo         Build it once with:
echo             python -m venv .venv
echo             .venv\Scripts\python.exe -m pip install -r requirements.txt
pause
exit /b 1

:no_code
echo [ERROR] tools\gui.py not found. Run this file from the project root.
pause
exit /b 1
