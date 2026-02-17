@echo off
chcp 65001 >nul
echo ========================================
echo    راهنمای اجرای پروژه GraphRAG
echo ========================================
echo.

REM بررسی Python
echo [1/6] بررسی نصب Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ✗ Python پیدا نشد! لطفاً Python را نصب کنید.
    pause
    exit /b 1
)
python --version
echo ✓ Python نصب شده است
echo.

REM بررسی pip
echo [2/6] بررسی نصب pip...
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ✗ pip پیدا نشد!
    pause
    exit /b 1
)
echo ✓ pip نصب شده است
echo.

REM بررسی requirements.txt
echo [3/6] بررسی فایل requirements.txt...
if not exist "requirements.txt" (
    echo ✗ فایل requirements.txt پیدا نشد!
    pause
    exit /b 1
)
echo ✓ فایل requirements.txt پیدا شد
echo.

REM بررسی محیط مجازی
echo [4/6] بررسی محیط مجازی...
if exist "graphrag_env\Scripts\activate.bat" (
    echo ✓ محیط مجازی پیدا شد
    echo فعال‌سازی محیط مجازی...
    call graphrag_env\Scripts\activate.bat
) else (
    echo ⚠ محیط مجازی پیدا نشد
    set /p create="آیا می‌خواهید محیط مجازی ایجاد شود؟ (y/n): "
    if /i "%create%"=="y" (
        echo ایجاد محیط مجازی...
        python -m venv graphrag_env
        call graphrag_env\Scripts\activate.bat
    )
)
echo.

REM بررسی وابستگی‌ها
echo [5/6] بررسی نصب وابستگی‌ها...
python -c "import flask" >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠ وابستگی‌ها نصب نشده‌اند
    set /p install="آیا می‌خواهید وابستگی‌ها را نصب کنید؟ (y/n): "
    if /i "%install%"=="y" (
        echo نصب وابستگی‌ها...
        pip install -r requirements.txt
        echo دانلود مدل spaCy...
        python -m spacy download en_core_web_sm
    )
) else (
    echo ✓ وابستگی‌ها نصب شده‌اند
)
echo.

REM بررسی فایل گراف
echo [6/6] بررسی فایل گراف...
if exist "..\hetionet_graph.pkl" (
    echo ✓ فایل گراف پیدا شد
) else if exist "hetionet_graph.pkl" (
    echo ✓ فایل گراف پیدا شد
) else (
    echo ⚠ فایل گراف پیدا نشد
    echo سیستم به صورت خودکار گراف را می‌سازد
)
echo.

echo ========================================
echo    آماده برای اجرا!
echo ========================================
echo.
echo روش‌های اجرا:
echo 1. رابط وب (توصیه می‌شود) - http://localhost:5000
echo 2. نسخه ساده (دمو)
echo 3. خروج
echo.

set /p choice="لطفاً یک گزینه انتخاب کنید (1-3): "

if "%choice%"=="1" (
    echo.
    echo اجرای رابط وب...
    echo بعد از اجرا، مرورگر را باز کنید و به آدرس زیر بروید:
    echo http://localhost:5000
    echo.
    echo برای توقف، Ctrl+C را فشار دهید
    echo.
    python web_app.py
) else if "%choice%"=="2" (
    echo.
    echo اجرای نسخه ساده...
    echo.
    python run_graphrag.py
) else if "%choice%"=="3" (
    echo خروج...
    exit /b 0
) else (
    echo گزینه نامعتبر!
    pause
    exit /b 1
)

pause
