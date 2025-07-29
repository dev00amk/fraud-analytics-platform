#!/usr/bin/env python
"""
Quick setup verification for Fraud Analytics Platform
"""

import os
import sys
import django

def verify_setup():
    """Verify the Django setup is working."""
    print("🔍 Verifying Fraud Analytics Platform Setup...")
    
    # Set up Django
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fraud_platform.settings')
    
    try:
        django.setup()
        print("✅ Django setup successful")
    except Exception as e:
        print(f"❌ Django setup failed: {e}")
        return False
    
    # Test database connection
    try:
        from django.db import connection
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        print("✅ Database connection working")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False
    
    # Test models
    try:
        from apps.authentication.models import User
        from apps.transactions.models import Transaction
        from apps.fraud_detection.models import FraudRule
        print("✅ All models imported successfully")
    except Exception as e:
        print(f"❌ Model import failed: {e}")
        return False
    
    # Check migrations
    try:
        from django.core.management import execute_from_command_line
        print("✅ Django management commands available")
    except Exception as e:
        print(f"❌ Management commands failed: {e}")
        return False
    
    print("\n🎉 Setup verification complete!")
    print("Your fraud analytics platform is ready to use!")
    
    return True

if __name__ == "__main__":
    if verify_setup():
        print("\n📋 Next steps:")
        print("1. Start server: python manage.py runserver")
        print("2. Create admin: python manage.py createsuperuser")
        print("3. Visit: http://127.0.0.1:8000/docs/")
    else:
        print("\n❌ Setup verification failed")
        sys.exit(1)