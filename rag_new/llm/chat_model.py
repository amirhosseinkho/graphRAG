# -*- coding: utf-8 -*-
"""
Chat Model - مدل چت LLM
"""
import json
import time
import logging
from typing import Dict, Any, List, Optional
import openai
import anthropic
from google.generativeai import GenerativeModel
import requests

class Base:
    """کلاس پایه برای مدل‌های LLM"""
    
    def __init__(self, model_name: str = None, api_key: str = None, base_url: str = None):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = 600
        self.max_retries = 3
        
    def _classify_error(self, error: Exception) -> str:
        """طبقه‌بندی خطاها"""
        error_str = str(error).lower()
        if "rate limit" in error_str or "quota" in error_str:
            return "rate_limit"
        elif "timeout" in error_str or "time out" in error_str:
            return "timeout"
        elif "invalid" in error_str or "bad request" in error_str:
            return "invalid_request"
        else:
            return "unknown"
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """تلاش مجدد با تاخیر تصاعدی"""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_type = self._classify_error(e)
                if error_type == "rate_limit":
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                elif error_type == "timeout":
                    if attempt < self.max_retries - 1:
                        time.sleep(1)
                    else:
                        raise
                else:
                    raise
        raise Exception("Max retries exceeded")
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """متد اصلی چت - باید در کلاس‌های فرزند پیاده‌سازی شود"""
        raise NotImplementedError

class GptTurbo(Base):
    """مدل GPT از OpenAI"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4o", base_url: str = "https://api.openai.com/v1"):
        super().__init__(model_name, api_key, base_url)
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            raise

class MoonshotChat(Base):
    """مدل Moonshot"""
    
    def __init__(self, api_key: str, model_name: str = "moonshot-v1-8k", base_url: str = "https://api.moonshot.cn/v1"):
        super().__init__(model_name, api_key, base_url)
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Moonshot API error: {e}")
            raise

class AzureChat(Base):
    """مدل Azure OpenAI"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4", base_url: str = None):
        super().__init__(model_name, api_key, base_url)
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Azure API error: {e}")
            raise

class QWenChat(Base):
    """مدل QWen از Alibaba"""
    
    def __init__(self, api_key: str, model_name: str = "qwen-turbo", base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        super().__init__(model_name, api_key, base_url)
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"QWen API error: {e}")
            raise

class ZhipuChat(Base):
    """مدل Zhipu"""
    
    def __init__(self, api_key: str, model_name: str = "glm-4", base_url: str = "https://open.bigmodel.cn/api/paas/v4"):
        super().__init__(model_name, api_key, base_url)
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Zhipu API error: {e}")
            raise

class OllamaChat(Base):
    """مدل Ollama محلی"""
    
    def __init__(self, model_name: str = "llama2", base_url: str = "http://localhost:11434"):
        super().__init__(model_name, None, base_url)
        self.base_url = base_url
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        try:
            url = f"{self.base_url}/v1/chat/completions"
            payload = {
                "model": self.model_name,
                "messages": messages,
                **kwargs
            }
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logging.error(f"Ollama API error: {e}")
            raise

class GeminiChat(Base):
    """مدل Gemini از Google"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        super().__init__(model_name, api_key)
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = GenerativeModel(model_name)
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        try:
            # تبدیل پیام‌ها به فرمت Gemini
            prompt = ""
            for msg in messages:
                if msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {msg['content']}\n"
            
            response = self.model.generate_content(prompt, **kwargs)
            return response.text
        except Exception as e:
            logging.error(f"Gemini API error: {e}")
            raise

class AnthropicChat(Base):
    """مدل Claude از Anthropic"""
    
    def __init__(self, api_key: str, model_name: str = "claude-3-sonnet-20240229"):
        super().__init__(model_name, api_key)
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        try:
            # تبدیل پیام‌ها به فرمت Anthropic
            system_message = ""
            user_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                elif msg["role"] == "user":
                    user_messages.append(msg["content"])
                elif msg["role"] == "assistant":
                    # Anthropic از فرمت متفاوتی استفاده می‌کند
                    pass
            
            user_content = "\n".join(user_messages)
            
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=kwargs.get("max_tokens", 1000),
                system=system_message,
                messages=[{"role": "user", "content": user_content}]
            )
            return response.content[0].text
        except Exception as e:
            logging.error(f"Anthropic API error: {e}")
            raise 