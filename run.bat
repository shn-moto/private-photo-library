@echo off
REM Wrapper для запуска скриптов с правильным PYTHONPATH

setlocal enabledelayedexpansion

REM Получить директорию скрипта
for %%i in ("%~f0") do set "SCRIPT_DIR=%%~dpi"

REM Установить PYTHONPATH
set PYTHONPATH=%SCRIPT_DIR%..;%PYTHONPATH%

REM Запустить Python скрипт
python %*

endlocal
