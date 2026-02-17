# اسکریپت نصب وابستگی‌های مورد نیاز برای پروژه (PowerShell)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "نصب وابستگی‌های پروژه" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# فعال‌سازی محیط مجازی
Write-Host "فعال‌سازی محیط مجازی..." -ForegroundColor Yellow
conda activate graphrag_env
if ($LASTEXITCODE -ne 0) {
    Write-Host "خطا: محیط مجازی graphrag_env پیدا نشد!" -ForegroundColor Red
    Write-Host "لطفاً ابتدا محیط مجازی را ایجاد کنید:" -ForegroundColor Yellow
    Write-Host "conda create -n graphrag_env python=3.10" -ForegroundColor Yellow
    Read-Host "برای خروج Enter را فشار دهید"
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "نصب کتابخانه‌های پایه" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "نصب کتابخانه‌های فارسی" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
pip install hazm
pip install fa-spellchecker

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "نصب neuralcoref (اختیاری - برای انگلیسی)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "توجه: neuralcoref نیاز به spacy>=2.1.0 دارد" -ForegroundColor Yellow
pip install neuralcoref

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "دانلود مدل‌های spaCy" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
python -m spacy download en_core_web_sm
python -m spacy download fa_core_news_sm

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "نصب کامل شد!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "برای استفاده از قابلیت‌های جدید:" -ForegroundColor Yellow
Write-Host "1. مدل‌های HuggingFace به صورت خودکار دانلود می‌شوند" -ForegroundColor White
Write-Host "2. برای استفاده از مدل‌های HuggingFace، توکن خود را از" -ForegroundColor White
Write-Host "   https://huggingface.co/settings/tokens دریافت کنید" -ForegroundColor White
Write-Host ""
Read-Host "برای خروج Enter را فشار دهید"
