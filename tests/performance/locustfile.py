"""
Performance and load testing for fraud analytics platform.
Enterprise-grade load testing with Locust.
"""

import json
import random
from datetime import datetime, timedelta

from locust import HttpUser, between, task


class FraudAnalyticsUser(HttpUser):
    """Simulated user for load testing fraud analytics platform."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Set up user session."""
        # Register and authenticate user
        self.register_user()
        self.authenticate()
    
    def register_user(self):
        """Register a test user."""
        user_id = random.randint(1000, 9999)
        self.user_data = {
            "username": f"loadtest_user_{user_id}",
            "email": f"loadtest_{user_id}@example.com",
            "password": "loadtest123",
            "company_name": f"LoadTest Company {user_id}"
        }
        
        response = self.client.post("/api/v1/auth/register/", json=self.user_data)
        if response.status_code == 201:
            print(f"✅ User {self.user_data['username']} registered successfully")
    
    def authenticate(self):
        """Authenticate user and get JWT token."""
        login_data = {
            "email": self.user_data["email"],
            "password": self.user_data["password"]
        }
        
        response = self.client.post("/api/v1/auth/token/", json=login_data)
        if response.status_code == 200:
            token_data = response.json()
            self.token = token_data["access"]
            self.client.headers.update({"Authorization": f"Bearer {self.token}"})
            print(f"✅ User {self.user_data['username']} authenticated")
    
    @task(5)
    def analyze_transaction(self):
        """Test fraud analysis endpoint (most frequent operation)."""
        transaction_data = self.generate_transaction_data()
        
        with self.client.post(
            "/api/v1/fraud/analyze/",
            json=transaction_data,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if "fraud_score" in result and "risk_level" in result:
                    response.success()
                else:
                    response.failure("Missing required fields in response")
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(3)
    def create_transaction(self):
        """Test transaction creation."""
        transaction_data = self.generate_transaction_data()
        
        with self.client.post(
            "/api/v1/transactions/",
            json=transaction_data,
            catch_response=True
        ) as response:
            if response.status_code == 201:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(2)
    def get_transactions(self):
        """Test transaction listing."""
        with self.client.get(
            "/api/v1/transactions/",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "results" in data:
                    response.success()
                else:
                    response.failure("Missing results in response")
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(2)
    def get_dashboard_stats(self):
        """Test dashboard statistics."""
        with self.client.get(
            "/api/v1/analytics/dashboard/",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                required_fields = [
                    "total_transactions",
                    "flagged_transactions",
                    "open_cases",
                    "average_fraud_score"
                ]
                if all(field in data for field in required_fields):
                    response.success()
                else:
                    response.failure("Missing required fields in dashboard stats")
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(1)
    def create_fraud_rule(self):
        """Test fraud rule creation."""
        rule_data = {
            "name": f"Load Test Rule {random.randint(1000, 9999)}",
            "description": "Automated load test rule",
            "conditions": {"amount_threshold": random.randint(100, 1000)},
            "action": random.choice(["flag", "alert", "decline"]),
            "priority": random.randint(1, 10)
        }
        
        with self.client.post(
            "/api/v1/fraud/rules/",
            json=rule_data,
            catch_response=True
        ) as response:
            if response.status_code == 201:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(1)
    def create_case(self):
        """Test case creation."""
        case_data = {
            "title": f"Load Test Case {random.randint(1000, 9999)}",
            "description": "Automated load test case",
            "transaction_id": f"txn_load_{random.randint(10000, 99999)}",
            "priority": random.choice(["low", "medium", "high", "critical"])
        }
        
        with self.client.post(
            "/api/v1/cases/",
            json=case_data,
            catch_response=True
        ) as response:
            if response.status_code == 201:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(1)
    def get_fraud_alerts(self):
        """Test fraud alerts listing."""
        with self.client.get(
            "/api/v1/fraud/alerts/",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")
    
    def generate_transaction_data(self):
        """Generate realistic transaction data for testing."""
        merchants = [
            "amazon_store", "walmart_online", "target_retail",
            "bestbuy_electronics", "starbucks_coffee", "mcdonalds_food",
            "shell_gas", "uber_rides", "netflix_streaming", "spotify_music"
        ]
        
        payment_methods = ["credit_card", "debit_card", "paypal", "apple_pay", "google_pay"]
        currencies = ["USD", "EUR", "GBP", "CAD", "AUD"]
        
        # Generate realistic amounts with some high-value transactions
        amount_ranges = [
            (5.00, 50.00, 0.4),      # Small transactions (40%)
            (50.00, 200.00, 0.3),    # Medium transactions (30%)
            (200.00, 500.00, 0.2),   # Large transactions (20%)
            (500.00, 2000.00, 0.08), # Very large transactions (8%)
            (2000.00, 10000.00, 0.02) # Suspicious amounts (2%)
        ]
        
        # Select amount range based on probability
        rand = random.random()
        cumulative = 0
        for min_amt, max_amt, prob in amount_ranges:
            cumulative += prob
            if rand <= cumulative:
                amount = round(random.uniform(min_amt, max_amt), 2)
                break
        else:
            amount = round(random.uniform(5.00, 50.00), 2)
        
        # Generate IP addresses (some suspicious)
        ip_ranges = [
            "192.168.1.{}", "10.0.0.{}", "172.16.0.{}",  # Normal IPs
            "203.0.113.{}", "198.51.100.{}"              # Suspicious IPs
        ]
        ip_template = random.choice(ip_ranges)
        ip_address = ip_template.format(random.randint(1, 254))
        
        # Generate timestamp (recent transactions)
        now = datetime.now()
        time_offset = timedelta(
            hours=random.randint(0, 24),
            minutes=random.randint(0, 59)
        )
        timestamp = (now - time_offset).isoformat() + "Z"
        
        return {
            "transaction_id": f"txn_load_{random.randint(100000, 999999)}",
            "user_id": f"user_{random.randint(1000, 9999)}",
            "amount": amount,
            "currency": random.choice(currencies),
            "merchant_id": random.choice(merchants),
            "payment_method": random.choice(payment_methods),
            "ip_address": ip_address,
            "user_agent": "LoadTest/1.0 (Performance Testing)",
            "timestamp": timestamp
        }


class HighVolumeUser(HttpUser):
    """High-volume user for stress testing."""
    
    wait_time = between(0.1, 0.5)  # Very fast requests
    
    def on_start(self):
        """Set up high-volume user."""
        self.register_user()
        self.authenticate()
    
    def register_user(self):
        """Register a high-volume test user."""
        user_id = random.randint(10000, 99999)
        self.user_data = {
            "username": f"hvuser_{user_id}",
            "email": f"hvuser_{user_id}@example.com",
            "password": "hvtest123",
            "company_name": f"HighVolume Corp {user_id}"
        }
        
        self.client.post("/api/v1/auth/register/", json=self.user_data)
    
    def authenticate(self):
        """Authenticate high-volume user."""
        login_data = {
            "email": self.user_data["email"],
            "password": self.user_data["password"]
        }
        
        response = self.client.post("/api/v1/auth/token/", json=login_data)
        if response.status_code == 200:
            token_data = response.json()
            self.token = token_data["access"]
            self.client.headers.update({"Authorization": f"Bearer {self.token}"})
    
    @task
    def rapid_fraud_analysis(self):
        """Rapid fraud analysis for stress testing."""
        transaction_data = {
            "transaction_id": f"hvtxn_{random.randint(1000000, 9999999)}",
            "user_id": f"hvuser_{random.randint(1000, 9999)}",
            "amount": round(random.uniform(10.00, 1000.00), 2),
            "currency": "USD",
            "merchant_id": "stress_test_merchant",
            "payment_method": "credit_card",
            "ip_address": f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
            "timestamp": datetime.now().isoformat() + "Z"
        }
        
        self.client.post("/api/v1/fraud/analyze/", json=transaction_data)


class APIHealthCheckUser(HttpUser):
    """User for health check and monitoring endpoints."""
    
    wait_time = between(5, 10)  # Less frequent health checks
    
    @task
    def health_check(self):
        """Test health check endpoint."""
        with self.client.get("/health/", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    response.success()
                else:
                    response.failure("Health check returned unhealthy status")
            else:
                response.failure(f"Health check failed with status {response.status_code}")
    
    @task
    def api_docs(self):
        """Test API documentation endpoint."""
        with self.client.get("/docs/", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"API docs failed with status {response.status_code}")


# Custom load test scenarios
class SpikeTestUser(FraudAnalyticsUser):
    """User for spike testing - sudden load increases."""
    
    wait_time = between(0.1, 1.0)  # Faster than normal users
    
    @task(10)  # Much higher weight for fraud analysis
    def spike_fraud_analysis(self):
        """High-frequency fraud analysis for spike testing."""
        self.analyze_transaction()


class SoakTestUser(FraudAnalyticsUser):
    """User for soak testing - sustained load over time."""
    
    wait_time = between(2, 5)  # Consistent moderate load
    
    def on_start(self):
        """Set up soak test user."""
        super().on_start()
        print(f"🔄 Soak test user {self.user_data['username']} started")
    
    def on_stop(self):
        """Clean up soak test user."""
        print(f"🛑 Soak test user {self.user_data['username']} stopped")


# Performance test configuration
class WebsiteUser(HttpUser):
    """Combined user class for comprehensive testing."""
    
    # Weight distribution for different user types
    tasks = {
        FraudAnalyticsUser: 70,    # 70% normal users
        HighVolumeUser: 20,        # 20% high-volume users
        APIHealthCheckUser: 10     # 10% monitoring users
    }
    
    wait_time = between(1, 5)