# اسکریپت سریع برای حل warning ها (PowerShell)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "حل Warning های سیستم" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# فعال‌سازی محیط مجازی
conda activate graphrag_env

Write-Host "نصب کتابخانه‌های فارسی..." -ForegroundColor Yellow
pip install hazm fa-spellchecker

Write-Host ""
Write-Host "دانلود مدل spaCy فارسی..." -ForegroundColor Yellow
python -m spacy download fa_core_news_sm

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "تکمیل شد!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "توجه: خطای Redis یک warning است و سیستم به درستی کار می‌کند." -ForegroundColor Yellow
Write-Host ""
Read-Host "برای خروج Enter را فشار دهید"
