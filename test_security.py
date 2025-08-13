#!/usr/bin/env python
"""
Basic security framework functionality test
"""

import os
import sys
import django
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fraud_platform.settings')
django.setup()

def test_security_imports():
    """Test that all security components can be imported."""
    try:
        from apps.security import (
            security_manager,
            auth_manager,
            authz_manager,
            security_monitor,
            audit_logger,
            input_validator,
            config_manager
        )
        print("+ All security components imported successfully")
        return True
    except ImportError as e:
        print(f"X Import error: {e}")
        return False

def test_security_initialization():
    """Test that security components initialize correctly."""
    try:
        from apps.security import security_manager
        
        # Test token generation
        token = security_manager.generate_secure_token(16)
        assert len(token) >= 16, "Token generation failed"
        
        # Test password hashing
        password = "test_password_123"
        hashed = security_manager.hash_password(password)
        assert security_manager.verify_password(password, hashed), "Password verification failed"
        
        print("+ Security manager functionality verified")
        return True
    except Exception as e:
        print(f"X Security initialization error: {e}")
        return False

def test_input_validation():
    """Test input validation functionality."""
    try:
        from apps.security import input_validator
        
        # Test basic string validation
        schema = {
            'username': {
                'type': 'string',
                'required': True,
                'min_length': 3,
                'max_length': 50
            }
        }
        
        data = {'username': 'testuser'}
        validated = input_validator.validate_and_sanitize(data, schema)
        assert validated['username'] == 'testuser', "Basic validation failed"
        
        print("+ Input validation functionality verified")
        return True
    except Exception as e:
        print(f"X Input validation error: {e}")
        return False

def main():
    """Run all security tests."""
    print("Testing Security Framework...")
    print("=" * 50)
    
    tests = [
        test_security_imports,
        test_security_initialization,
        test_input_validation
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print("=" * 50)
    print(f"+ {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("Security framework is ready for deployment!")
        return True
    else:
        print("Some tests failed. Please review the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)