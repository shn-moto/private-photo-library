@echo off
chcp 65001 >nul
echo ========================================
echo Docker Cleanup Script
echo ========================================
echo.

echo [1/3] Проверка текущего состояния Docker...
docker system df
echo.

echo [2/3] Удаление неиспользуемых данных...
echo - Удаляются: неиспользуемые образы, контейнеры, build cache
docker system prune -a -f
echo.

echo [3/3] Удаление неиспользуемых volumes...
docker volume prune -f
echo.

echo ========================================
echo Финальное состояние:
echo ========================================
docker system df
echo.

echo ✓ Очистка завершена!
pause
