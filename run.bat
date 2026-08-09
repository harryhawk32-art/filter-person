@echo off
chcp 932 > nul
echo ========================================
echo   【画像整理】顔が写っていない画像を削除するツール 
echo ========================================

python "%~dp0fp.py"

if %ERRORLEVEL% EQU 1 (
    echo.
    cls
) else goto :end_of_script

:end_of_script
echo "処理終了しました。" > nul
