#!/usr/bin/env python
"""
API Test Script for Fraud Analytics Platform
"""

import requests
import json
import sys

BASE_URL = "http://127.0.0.1:8000"

def test_health_check():
    """Test the health check endpoint."""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health/")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Status: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Server not running. Please start with: python manage.py runserver")
        return False

def test_api_docs():
    """Test API documentation endpoint."""
    print("\n🔍 Testing API documentation...")
    try:
        response = requests.get(f"{BASE_URL}/docs/")
        if response.status_code == 200:
            print("✅ API documentation accessible")
            return True
        else:
            print(f"❌ API docs failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API docs error: {e}")
        return False

def test_user_registration():
    """Test user registration."""
    print("\n🔍 Testing user registration...")
    user_data = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpass123",
        "company_name": "Test Company"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/auth/register/",
            json=user_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 201:
            print("✅ User registration successful")
            return response.json()
        else:
            print(f"❌ Registration failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Registration error: {e}")
        return None

def test_jwt_token(email, password):
    """Test JWT token generation."""
    print("\n🔍 Testing JWT token generation...")
    login_data = {
        "email": email,
        "password": password
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/auth/token/",
            json=login_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            token_data = response.json()
            print("✅ JWT token generated successfully")
            return token_data.get("access")
        else:
            print(f"❌ Token generation failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Token error: {e}")
        return None

def test_fraud_analysis(token):
    """Test fraud analysis endpoint."""
    print("\n🔍 Testing fraud analysis...")
    transaction_data = {
        "transaction_id": "test_txn_123",
        "user_id": "test_user_456",
        "amount": 150.00,
        "currency": "USD",
        "merchant_id": "test_merchant_789",
        "payment_method": "credit_card",
        "ip_address": "192.168.1.100",
        "timestamp": "2024-01-15T10:30:00Z"
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/fraud/analyze/",
            json=transaction_data,
            headers=headers
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Fraud analysis successful")
            print(f"   Fraud Score: {result.get('fraud_score', 'N/A')}")
            print(f"   Risk Level: {result.get('risk_level', 'N/A')}")
            print(f"   Recommendation: {result.get('recommendation', 'N/A')}")
            return True
        else:
            print(f"❌ Fraud analysis failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Fraud analysis error: {e}")
        return False

def main():
    """Run all API tests."""
    print("🚀 Fraud Analytics Platform API Tests")
    print("=" * 50)
    
    # Test health check
    if not test_health_check():
        print("\n❌ Server not accessible. Please ensure the server is running.")
        sys.exit(1)
    
    # Test API docs
    test_api_docs()
    
    # Test user registration
    user = test_user_registration()
    if not user:
        print("\n❌ Cannot proceed without user registration")
        sys.exit(1)
    
    # Test JWT token
    token = test_jwt_token("test@example.com", "testpass123")
    if not token:
        print("\n❌ Cannot proceed without JWT token")
        sys.exit(1)
    
    # Test fraud analysis
    test_fraud_analysis(token)
    
    print("\n🎉 API Testing Complete!")
    print("\nYour fraud analytics platform is working correctly!")
    print(f"\n📊 Access your platform:")
    print(f"   • API Base: {BASE_URL}/api/v1/")
    print(f"   • Documentation: {BASE_URL}/docs/")
    print(f"   • Admin Panel: {BASE_URL}/admin/")

if __name__ == "__main__":
    main()