@echo off
REM اسکریپت نصب وابستگی‌های مورد نیاز برای پروژه
chcp 65001 >nul
echo ========================================
echo نصب وابستگی‌های پروژه
echo ========================================
echo.

REM فعال‌سازی محیط مجازی
echo فعال‌سازی محیط مجازی...
call conda activate graphrag_env
if errorlevel 1 (
    echo خطا: محیط مجازی graphrag_env پیدا نشد!
    echo لطفاً ابتدا محیط مجازی را ایجاد کنید:
    echo conda create -n graphrag_env python=3.10
    pause
    exit /b 1
)

echo.
echo ========================================
echo نصب کتابخانه‌های پایه
echo ========================================
pip install -r requirements.txt

echo.
echo ========================================
echo نصب کتابخانه‌های فارسی
echo ========================================
pip install hazm
pip install fa-spellchecker

echo.
echo ========================================
echo نصب neuralcoref (اختیاری - برای انگلیسی)
echo ========================================
echo توجه: neuralcoref نیاز به spacy>=2.1.0 دارد
pip install neuralcoref

echo.
echo ========================================
echo دانلود مدل‌های spaCy
echo ========================================
python -m spacy download en_core_web_sm
python -m spacy download fa_core_news_sm

echo.
echo ========================================
echo نصب کامل شد!
echo ========================================
echo.
echo برای استفاده از قابلیت‌های جدید:
echo 1. مدل‌های HuggingFace به صورت خودکار دانلود می‌شوند
echo 2. برای استفاده از مدل‌های HuggingFace، توکن خود را از
echo    https://huggingface.co/settings/tokens دریافت کنید
echo.
pause
