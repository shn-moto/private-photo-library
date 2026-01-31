@echo off
REM Бэкап базы данных smart_photo_index
REM Использование: backup_db.bat

setlocal

REM Настройки подключения (из .env)
set PGHOST=localhost
set PGPORT=5432
set PGUSER=dev
set PGPASSWORD=secret
set PGDATABASE=smart_photo_index

REM Папка для бэкапов
set BACKUP_DIR=%~dp0..\backups
if not exist "%BACKUP_DIR%" mkdir "%BACKUP_DIR%"

REM Имя файла с датой и временем
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set TIMESTAMP=%datetime:~0,8%_%datetime:~8,6%
set BACKUP_FILE=%BACKUP_DIR%\smart_photo_index_%TIMESTAMP%.sql

echo Создание бэкапа базы данных...
echo Файл: %BACKUP_FILE%

REM Выполнить pg_dump
docker exec smart_photo_db pg_dump -U %PGUSER% -d %PGDATABASE% --no-owner --no-acl > "%BACKUP_FILE%"

if %ERRORLEVEL% EQU 0 (
    echo Бэкап успешно создан: %BACKUP_FILE%

    REM Показать размер файла
    for %%A in ("%BACKUP_FILE%") do echo Размер: %%~zA bytes
) else (
    echo ОШИБКА: Не удалось создать бэкап
    echo Проверьте, что Docker контейнер smart_photo_db запущен
)

endlocal
