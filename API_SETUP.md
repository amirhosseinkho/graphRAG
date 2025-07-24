# 🔑 راهنمای تنظیم API Key ها

## 📋 مدل‌های رایگان (بدون نیاز به API Key)

### 🤗 HuggingFace Models
- **وضعیت:** رایگان ✅
- **کیفیت:** خوب
- **سرعت:** متوسط
- **نیاز به نصب:** `pip install transformers torch`

## 💰 مدل‌های پولی (نیاز به API Key)

### 🚀 OpenAI GPT
- **وضعیت:** پولی (اعتبار رایگان محدود) ✅ **فعال**
- **کیفیت:** عالی
- **سرعت:** سریع
- **قیمت:** ~$0.002 per 1K tokens
- **API Key:** تنظیم شده

**نحوه دریافت API Key:**
1. به [OpenAI Platform](https://platform.openai.com/) بروید
2. حساب کاربری ایجاد کنید
3. در بخش API Keys، کلید جدید ایجاد کنید
4. کلید را کپی کنید

**تنظیم در کد:**
```python
service = GraphRAGService()
service.set_openai_api_key("your-api-key-here")
```

### 🧠 Anthropic Claude
- **وضعیت:** پولی
- **کیفیت:** عالی
- **سرعت:** سریع
- **قیمت:** ~$0.003 per 1K tokens

**نحوه دریافت API Key:**
1. به [Anthropic Console](https://console.anthropic.com/) بروید
2. حساب کاربری ایجاد کنید
3. در بخش API Keys، کلید جدید ایجاد کنید
4. کلید را کپی کنید

**تنظیم در کد:**
```python
service = GraphRAGService()
service.set_anthropic_api_key("your-api-key-here")
```

### 🌟 Google Gemini
- **وضعیت:** پولی (اعتبار رایگان محدود)
- **کیفیت:** عالی
- **سرعت:** سریع
- **قیمت:** ~$0.001 per 1K tokens

**نحوه دریافت API Key:**
1. به [Google AI Studio](https://makersuite.google.com/app/apikey) بروید
2. با حساب Google وارد شوید
3. API Key جدید ایجاد کنید
4. کلید را کپی کنید

**تنظیم در کد:**
```python
service = GraphRAGService()
service.set_gemini_api_key("your-api-key-here")
```

## 🔧 نحوه استفاده در برنامه وب

### روش 1: تنظیم مستقیم در کد
```python
# در فایل web_app.py
graphrag_service = GraphRAGService()

# تنظیم API Key ها
graphrag_service.set_openai_api_key("your-openai-key")
graphrag_service.set_anthropic_api_key("your-claude-key")
graphrag_service.set_gemini_api_key("your-gemini-key")
```

### روش 2: استفاده از متغیرهای محیطی
```python
import os
from dotenv import load_dotenv

load_dotenv()

graphrag_service = GraphRAGService()

# تنظیم از متغیرهای محیطی
if os.getenv('OPENAI_API_KEY'):
    graphrag_service.set_openai_api_key(os.getenv('OPENAI_API_KEY'))

if os.getenv('ANTHROPIC_API_KEY'):
    graphrag_service.set_anthropic_api_key(os.getenv('ANTHROPIC_API_KEY'))

if os.getenv('GEMINI_API_KEY'):
    graphrag_service.set_gemini_api_key(os.getenv('GEMINI_API_KEY'))
```

### فایل .env
```env
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-claude-api-key-here
GEMINI_API_KEY=your-gemini-api-key-here
```

## 📊 مقایسه مدل‌ها

| مدل | کیفیت | سرعت | هزینه | زبان فارسی |
|-----|--------|-------|-------|------------|
| HuggingFace | خوب | متوسط | رایگان | محدود |
| OpenAI GPT | عالی | سریع | پولی | خوب |
| Claude | عالی | سریع | پولی | عالی |
| Gemini | عالی | سریع | پولی | خوب |

## 💡 توصیه‌ها

1. **برای شروع:** از HuggingFace استفاده کنید (رایگان)
2. **برای کیفیت بهتر:** OpenAI GPT یا Claude
3. **برای هزینه کمتر:** Gemini
4. **برای زبان فارسی:** Claude بهترین گزینه است

## ⚠️ نکات مهم

- API Key ها را در کد قرار ندهید
- از فایل .env استفاده کنید
- فایل .env را در .gitignore قرار دهید
- اعتبار رایگان را مدیریت کنید
- از Rate Limiting آگاه باشید 