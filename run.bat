@echo off
chcp 65001 > nul
title 顔人数フィルター

if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON=%~dp0.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

"%PYTHON%" "%~dp0fp.py" %*
set "RESULT=%ERRORLEVEL%"
echo.
if not "%RESULT%"=="0" echo エラーが発生しました。上のメッセージを確認してください。
pause
exit /b %RESULT%
