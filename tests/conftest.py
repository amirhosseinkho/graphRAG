# -*- coding: utf-8 -*-
"""
Pytest configuration for GraphRAG tests
"""

import sys
import os
from pathlib import Path

# اضافه کردن مسیر اصلی پروژه به sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# تنظیمات pytest
def pytest_configure(config):
    """تنظیمات pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )

def pytest_collection_modifyitems(config, items):
    """تغییر آیتم‌های تست"""
    for item in items:
        # علامت‌گذاری تست‌های کند
        if "comprehensive" in item.name or "final" in item.name:
            item.add_marker("slow")
        
        # علامت‌گذاری تست‌های یکپارچگی
        if "system" in item.name or "integration" in item.name:
            item.add_marker("integration")
        
        # علامت‌گذاری تست‌های واحد
        if "basic" in item.name or "simple" in item.name:
            item.add_marker("unit") 