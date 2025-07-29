#!/usr/bin/env python
"""
Simple test script to verify the fraud analytics platform setup.
"""

def test_imports():
    """Test basic imports."""
    try:
        import django
        print(f"✅ Django {django.get_version()} imported successfully")
    except ImportError as e:
        print(f"❌ Django import failed: {e}")
        return False
    
    try:
        import rest_framework
        print(f"✅ Django REST Framework imported successfully")
    except ImportError as e:
        print(f"❌ Django REST Framework import failed: {e}")
        return False
    
    try:
        import redis
        print(f"✅ Redis client imported successfully")
    except ImportError as e:
        print(f"❌ Redis import failed: {e}")
        return False
    
    return True

def test_django_setup():
    """Test Django configuration."""
    import os
    import django
    from django.conf import settings
    
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fraud_platform.settings')
    
    try:
        django.setup()
        print(f"✅ Django setup successful")
        print(f"✅ Database: {settings.DATABASES['default']['ENGINE']}")
        return True
    except Exception as e:
        print(f"❌ Django setup failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing Fraud Analytics Platform Setup...")
    print("=" * 50)
    
    if test_imports():
        print("\n📦 All imports successful!")
        
        if test_django_setup():
            print("\n🎉 Setup verification complete!")
            print("\nNext steps:")
            print("1. Run: python manage.py makemigrations")
            print("2. Run: python manage.py migrate")
            print("3. Run: python manage.py createsuperuser")
            print("4. Run: python manage.py runserver")
        else:
            print("\n❌ Django setup failed")
    else:
        print("\n❌ Import tests failed")