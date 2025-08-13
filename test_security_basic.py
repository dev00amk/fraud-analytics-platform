#!/usr/bin/env python
"""
Basic security framework structure test (no Redis required)
"""

import os
import sys
from pathlib import Path

def test_security_files():
    """Test that all security files exist."""
    base_path = Path(__file__).parent / 'apps' / 'security'
    
    required_files = [
        'security_manager.py',
        'api_security.py', 
        'database_security.py',
        'audit_monitoring.py',
        'input_validation.py',
        'config_management.py',
        'error_handling.py',
        '__init__.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not (base_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"X Missing security files: {', '.join(missing_files)}")
        return False
    
    print("+ All security files present")
    return True

def test_security_imports_syntax():
    """Test that security modules have valid Python syntax."""
    import ast
    base_path = Path(__file__).parent / 'apps' / 'security'
    
    files_to_check = [
        'security_manager.py',
        'api_security.py', 
        'database_security.py',
        'input_validation.py',
        'error_handling.py'
    ]
    
    for file in files_to_check:
        file_path = base_path / file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            ast.parse(code)
        except SyntaxError as e:
            print(f"X Syntax error in {file}: {e}")
            return False
        except Exception as e:
            print(f"X Error reading {file}: {e}")
            return False
    
    print("+ All security files have valid syntax")
    return True

def test_django_settings():
    """Test that Django settings include security configurations."""
    settings_path = Path(__file__).parent / 'fraud_platform' / 'settings.py'
    
    try:
        with open(settings_path, 'r', encoding='utf-8') as f:
            settings_content = f.read()
        
        required_configs = [
            'APISecurityMiddleware',
            'SecurityErrorMiddleware', 
            'SECURITY_FRAMEWORK_ENABLED',
            'REDIS_URL'
        ]
        
        missing_configs = []
        for config in required_configs:
            if config not in settings_content:
                missing_configs.append(config)
        
        if missing_configs:
            print(f"X Missing Django configurations: {', '.join(missing_configs)}")
            return False
        
        print("+ Django settings configured for security framework")
        return True
        
    except Exception as e:
        print(f"X Error checking Django settings: {e}")
        return False

def test_requirements():
    """Test that requirements include security dependencies."""
    requirements_path = Path(__file__).parent / 'requirements.txt'
    
    try:
        with open(requirements_path, 'r', encoding='utf-8') as f:
            requirements_content = f.read()
        
        required_deps = [
            'cryptography',
            'bcrypt',
            'PyJWT',
            'bleach'
        ]
        
        missing_deps = []
        for dep in required_deps:
            if dep not in requirements_content:
                missing_deps.append(dep)
        
        if missing_deps:
            print(f"X Missing dependencies: {', '.join(missing_deps)}")
            return False
        
        print("+ All security dependencies listed in requirements")
        return True
        
    except Exception as e:
        print(f"X Error checking requirements: {e}")
        return False

def main():
    """Run all basic security tests."""
    print("Testing Security Framework Structure...")
    print("=" * 50)
    
    tests = [
        test_security_files,
        test_security_imports_syntax,
        test_django_settings,
        test_requirements
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print("=" * 50)
    print(f"+ {passed}/{len(tests)} structure tests passed")
    
    if passed == len(tests):
        print("Security framework structure is valid!")
        print("Note: Full functionality testing requires Redis to be running.")
        return True
    else:
        print("Some structure tests failed. Please review the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)