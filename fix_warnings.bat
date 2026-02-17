@echo off
REM اسکریپت سریع برای حل warning ها
chcp 65001 >nul
echo ========================================
echo حل Warning های سیستم
echo ========================================
echo.

REM فعال‌سازی محیط مجازی
call conda activate graphrag_env

echo نصب کتابخانه‌های فارسی...
pip install hazm fa-spellchecker

echo.
echo دانلود مدل spaCy فارسی...
python -m spacy download fa_core_news_sm

echo.
echo ========================================
echo تکمیل شد!
echo ========================================
echo.
echo توجه: خطای Redis یک warning است و سیستم به درستی کار می‌کند.
echo.
pause
