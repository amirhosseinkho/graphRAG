# -*- coding: utf-8 -*-
"""
Redis Connection - اتصال به Redis
"""

import logging
import json
import pickle
from typing import Any, Optional
import redis

# تنظیمات پیش‌فرض Redis
REDIS_CONFIG = {
    "host": "localhost",
    "port": 6379,
    "db": 0,
    "password": None,
    "decode_responses": True,
    "socket_timeout": 5,
    "socket_connect_timeout": 5,
    "retry_on_timeout": True
}

class RedisConnection:
    """کلاس اتصال به Redis"""
    
    def __init__(self, **kwargs):
        self.config = {**REDIS_CONFIG, **kwargs}
        self.logger = logging.getLogger(__name__)
        self._client = None
    
    @property
    def client(self):
        """دریافت کلاینت Redis"""
        if self._client is None:
            try:
                self._client = redis.Redis(**self.config)
                # تست اتصال
                self._client.ping()
                self.logger.info("Redis connection established")
            except Exception as e:
                self.logger.error(f"Redis connection failed: {e}")
                self._client = None
        return self._client
    
    def get(self, key: str) -> Optional[Any]:
        """دریافت مقدار از Redis"""
        try:
            if self.client is None:
                return None
            
            value = self.client.get(key)
            if value is None:
                return None
            
            # تلاش برای deserialize
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                try:
                    return pickle.loads(value.encode('latin1'))
                except:
                    return value
                    
        except Exception as e:
            self.logger.error(f"Redis get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ex: int = None) -> bool:
        """ذخیره مقدار در Redis"""
        try:
            if self.client is None:
                return False
            
            # تلاش برای serialize
            try:
                serialized_value = json.dumps(value, ensure_ascii=False)
            except (TypeError, ValueError):
                try:
                    serialized_value = pickle.dumps(value).decode('latin1')
                except:
                    serialized_value = str(value)
            
            return self.client.set(key, serialized_value, ex=ex)
            
        except Exception as e:
            self.logger.error(f"Redis set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """حذف کلید از Redis"""
        try:
            if self.client is None:
                return False
            
            return bool(self.client.delete(key))
            
        except Exception as e:
            self.logger.error(f"Redis delete error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """بررسی وجود کلید"""
        try:
            if self.client is None:
                return False
            
            return bool(self.client.exists(key))
            
        except Exception as e:
            self.logger.error(f"Redis exists error: {e}")
            return False
    
    def expire(self, key: str, seconds: int) -> bool:
        """تنظیم زمان انقضا"""
        try:
            if self.client is None:
                return False
            
            return bool(self.client.expire(key, seconds))
            
        except Exception as e:
            self.logger.error(f"Redis expire error: {e}")
            return False
    
    def ttl(self, key: str) -> int:
        """دریافت زمان باقی‌مانده تا انقضا"""
        try:
            if self.client is None:
                return -1
            
            return self.client.ttl(key)
            
        except Exception as e:
            self.logger.error(f"Redis ttl error: {e}")
            return -1
    
    def keys(self, pattern: str = "*") -> list:
        """دریافت کلیدهای مطابق با الگو"""
        try:
            if self.client is None:
                return []
            
            return self.client.keys(pattern)
            
        except Exception as e:
            self.logger.error(f"Redis keys error: {e}")
            return []
    
    def flushdb(self) -> bool:
        """پاک کردن تمام دیتابیس"""
        try:
            if self.client is None:
                return False
            
            return bool(self.client.flushdb())
            
        except Exception as e:
            self.logger.error(f"Redis flushdb error: {e}")
            return False
    
    def info(self) -> dict:
        """دریافت اطلاعات Redis"""
        try:
            if self.client is None:
                return {}
            
            return self.client.info()
            
        except Exception as e:
            self.logger.error(f"Redis info error: {e}")
            return {}
    
    def ping(self) -> bool:
        """تست اتصال"""
        try:
            if self.client is None:
                return False
            
            return self.client.ping()
            
        except Exception as e:
            self.logger.error(f"Redis ping error: {e}")
            return False


class MockRedisConnection:
    """کلاس Mock Redis برای تست"""
    
    def __init__(self):
        self.data = {}
        self.expirations = {}
        self.logger = logging.getLogger(__name__)
    
    def get(self, key: str) -> Optional[Any]:
        """دریافت مقدار از Mock Redis"""
        if key not in self.data:
            return None
        
        # بررسی انقضا
        if key in self.expirations:
            import time
            if time.time() > self.expirations[key]:
                del self.data[key]
                del self.expirations[key]
                return None
        
        return self.data[key]
    
    def set(self, key: str, value: Any, ex: int = None) -> bool:
        """ذخیره مقدار در Mock Redis"""
        self.data[key] = value
        
        if ex is not None:
            import time
            self.expirations[key] = time.time() + ex
        
        return True
    
    def delete(self, key: str) -> bool:
        """حذف کلید از Mock Redis"""
        if key in self.data:
            del self.data[key]
            if key in self.expirations:
                del self.expirations[key]
            return True
        return False
    
    def exists(self, key: str) -> bool:
        """بررسی وجود کلید در Mock Redis"""
        return key in self.data
    
    def expire(self, key: str, seconds: int) -> bool:
        """تنظیم زمان انقضا در Mock Redis"""
        if key not in self.data:
            return False
        
        import time
        self.expirations[key] = time.time() + seconds
        return True
    
    def ttl(self, key: str) -> int:
        """دریافت زمان باقی‌مانده تا انقضا در Mock Redis"""
        if key not in self.data or key not in self.expirations:
            return -1
        
        import time
        remaining = self.expirations[key] - time.time()
        return max(0, int(remaining))
    
    def keys(self, pattern: str = "*") -> list:
        """دریافت کلیدهای مطابق با الگو در Mock Redis"""
        import fnmatch
        return [key for key in self.data.keys() if fnmatch.fnmatch(key, pattern)]
    
    def flushdb(self) -> bool:
        """پاک کردن تمام دیتابیس Mock Redis"""
        self.data.clear()
        self.expirations.clear()
        return True
    
    def info(self) -> dict:
        """دریافت اطلاعات Mock Redis"""
        return {
            "used_memory": len(self.data),
            "connected_clients": 1,
            "total_commands_processed": 0
        }
    
    def ping(self) -> bool:
        """تست اتصال Mock Redis"""
        return True


# ایجاد نمونه پیش‌فرض
try:
    REDIS_CONN = RedisConnection()
    # تست اتصال
    if not REDIS_CONN.ping():
        REDIS_CONN = MockRedisConnection()
        logging.warning("Using Mock Redis connection")
except Exception as e:
    REDIS_CONN = MockRedisConnection()
    logging.warning(f"Redis connection failed, using Mock: {e}") 