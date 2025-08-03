#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Runner for GraphRAG System
"""

import sys
import os
from pathlib import Path

# اضافه کردن مسیر اصلی به sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_specific_test(test_name: str):
    """اجرای یک تست خاص"""
    if test_name == "test_tp53_fix.py":
        # استفاده از test_runner برای تست TP53
        try:
            from tests.test_runner import run_tp53_test
            run_tp53_test()
            print("✅ تست TP53 با موفقیت اجرا شد!")
            return True
        except Exception as e:
            print(f"❌ خطا در اجرای تست TP53: {e}")
            return False
    elif test_name == "debug_tp53_retrieval.py":
        # استفاده از test_runner برای دیباگ TP53
        try:
            from tests.test_runner import run_debug_tp53
            run_debug_tp53()
            print("✅ دیباگ TP53 با موفقیت اجرا شد!")
            return True
        except Exception as e:
            print(f"❌ خطا در اجرای دیباگ TP53: {e}")
            return False
    else:
        # برای سایر تست‌ها، اجرای مستقیم
        test_file = f"tests/{test_name}"
        
        if not os.path.exists(test_file):
            print(f"❌ فایل تست {test_file} یافت نشد!")
            return False
        
        print(f"🧪 اجرای تست: {test_name}")
        print("=" * 50)
        
        try:
            # اضافه کردن مسیر اصلی به sys.path
            import sys
            from pathlib import Path
            project_root = Path(__file__).parent
            sys.path.insert(0, str(project_root))
            
            # اجرای تست
            exec(open(test_file).read())
            print("✅ تست با موفقیت اجرا شد!")
            return True
        except Exception as e:
            print(f"❌ خطا در اجرای تست: {e}")
            return False

def run_all_tests():
    """اجرای همه تست‌ها"""
    import glob
    
    test_files = glob.glob("tests/test_*.py")
    debug_files = glob.glob("tests/debug_*.py")
    all_files = test_files + debug_files
    
    print(f"🧪 یافت شد: {len(all_files)} فایل تست")
    print("=" * 50)
    
    success_count = 0
    total_count = len(all_files)
    
    for test_file in all_files:
        test_name = os.path.basename(test_file)
        print(f"\n🔍 اجرای: {test_name}")
        
        try:
            exec(open(test_file).read())
            print(f"✅ {test_name} - موفق")
            success_count += 1
        except Exception as e:
            print(f"❌ {test_name} - خطا: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 نتایج: {success_count}/{total_count} تست موفق")
    
    return success_count == total_count

def list_tests():
    """نمایش لیست تست‌ها"""
    import glob
    
    test_files = glob.glob("tests/test_*.py")
    debug_files = glob.glob("tests/debug_*.py")
    all_files = sorted(test_files + debug_files)
    
    print("📋 لیست تست‌های موجود:")
    print("=" * 50)
    
    for i, test_file in enumerate(all_files, 1):
        test_name = os.path.basename(test_file)
        print(f"{i:2d}. {test_name}")
    
    print(f"\n📊 مجموع: {len(all_files)} فایل تست")

def main():
    """تابع اصلی"""
    if len(sys.argv) < 2:
        print("🧪 GraphRAG Test Runner")
        print("=" * 30)
        print("استفاده:")
        print("  python run_tests.py list                    # نمایش لیست تست‌ها")
        print("  python run_tests.py all                     # اجرای همه تست‌ها")
        print("  python run_tests.py test_tp53_fix.py       # اجرای یک تست خاص")
        print("  python run_tests.py debug_tp53_retrieval.py # اجرای دیباگ")
        return
    
    command = sys.argv[1]
    
    if command == "list":
        list_tests()
    elif command == "all":
        success = run_all_tests()
        sys.exit(0 if success else 1)
    else:
        # اجرای تست خاص
        test_name = command
        if not test_name.endswith('.py'):
            test_name += '.py'
        
        success = run_specific_test(test_name)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 