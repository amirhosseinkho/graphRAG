# اسکریپت اجرای خودکار پروژه GraphRAG
# PowerShell Script for Running GraphRAG Project

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   راهنمای اجرای پروژه GraphRAG" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# بررسی Python
Write-Host "[1/6] بررسی نصب Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python نصب شده: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python پیدا نشد! لطفاً Python را نصب کنید." -ForegroundColor Red
    exit 1
}

# بررسی pip
Write-Host "[2/6] بررسی نصب pip..." -ForegroundColor Yellow
try {
    $pipVersion = pip --version 2>&1
    Write-Host "✓ pip نصب شده: $pipVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ pip پیدا نشد!" -ForegroundColor Red
    exit 1
}

# بررسی وجود requirements.txt
Write-Host "[3/6] بررسی فایل requirements.txt..." -ForegroundColor Yellow
if (Test-Path "requirements.txt") {
    Write-Host "✓ فایل requirements.txt پیدا شد" -ForegroundColor Green
} else {
    Write-Host "✗ فایل requirements.txt پیدا نشد!" -ForegroundColor Red
    exit 1
}

# بررسی محیط مجازی
Write-Host "[4/6] بررسی محیط مجازی..." -ForegroundColor Yellow
if (Test-Path "graphrag_env\Scripts\Activate.ps1") {
    Write-Host "✓ محیط مجازی پیدا شد" -ForegroundColor Green
    Write-Host "فعال‌سازی محیط مجازی..." -ForegroundColor Yellow
    & ".\graphrag_env\Scripts\Activate.ps1"
} else {
    Write-Host "⚠ محیط مجازی پیدا نشد" -ForegroundColor Yellow
    $create = Read-Host "آیا می‌خواهید محیط مجازی ایجاد شود؟ (y/n)"
    if ($create -eq "y" -or $create -eq "Y") {
        Write-Host "ایجاد محیط مجازی..." -ForegroundColor Yellow
        python -m venv graphrag_env
        Write-Host "فعال‌سازی محیط مجازی..." -ForegroundColor Yellow
        & ".\graphrag_env\Scripts\Activate.ps1"
    }
}

# بررسی نصب وابستگی‌ها
Write-Host "[5/6] بررسی نصب وابستگی‌ها..." -ForegroundColor Yellow
$checkPackage = python -c "import flask" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ وابستگی‌ها نصب شده‌اند" -ForegroundColor Green
} else {
    Write-Host "⚠ وابستگی‌ها نصب نشده‌اند" -ForegroundColor Yellow
    $install = Read-Host "آیا می‌خواهید وابستگی‌ها را نصب کنید؟ (y/n)"
    if ($install -eq "y" -or $install -eq "Y") {
        Write-Host "نصب وابستگی‌ها (این کار ممکن است چند دقیقه طول بکشد)..." -ForegroundColor Yellow
        pip install -r requirements.txt
        
        Write-Host "دانلود مدل spaCy..." -ForegroundColor Yellow
        python -m spacy download en_core_web_sm
    }
}

# بررسی فایل گراف
Write-Host "[6/6] بررسی فایل گراف..." -ForegroundColor Yellow
if (Test-Path "..\hetionet_graph.pkl") {
    Write-Host "✓ فایل گراف پیدا شد" -ForegroundColor Green
} elseif (Test-Path "hetionet_graph.pkl") {
    Write-Host "✓ فایل گراف پیدا شد" -ForegroundColor Green
} else {
    Write-Host "⚠ فایل گراف پیدا نشد" -ForegroundColor Yellow
    Write-Host "سیستم به صورت خودکار گراف را می‌سازد" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   آماده برای اجرا!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# انتخاب روش اجرا
Write-Host "روش‌های اجرا:" -ForegroundColor Yellow
Write-Host "1. رابط وب (توصیه می‌شود) - http://localhost:5000" -ForegroundColor White
Write-Host "2. نسخه ساده (دمو)" -ForegroundColor White
Write-Host "3. خروج" -ForegroundColor White
Write-Host ""

$choice = Read-Host "لطفاً یک گزینه انتخاب کنید (1-3)"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "اجرای رابط وب..." -ForegroundColor Green
        Write-Host "بعد از اجرا، مرورگر را باز کنید و به آدرس زیر بروید:" -ForegroundColor Yellow
        Write-Host "http://localhost:5000" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "برای توقف، Ctrl+C را فشار دهید" -ForegroundColor Yellow
        Write-Host ""
        python web_app.py
    }
    "2" {
        Write-Host ""
        Write-Host "اجرای نسخه ساده..." -ForegroundColor Green
        Write-Host ""
        python run_graphrag.py
    }
    "3" {
        Write-Host "خروج..." -ForegroundColor Yellow
        exit 0
    }
    default {
        Write-Host "گزینه نامعتبر!" -ForegroundColor Red
        exit 1
    }
}
