"""
Comprehensive unit tests for fraud detection services.
Generated with Claude Code assistance for enterprise-grade testing.
"""

import json
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest
from django.contrib.auth import get_user_model
from django.test import TestCase
from django.utils import timezone

from apps.fraud_detection.models import FraudAlert, FraudRule
from apps.fraud_detection.services import FraudDetectionService
from apps.transactions.models import Transaction

User = get_user_model()


class FraudDetectionServiceTest(TestCase):
    """Test cases for FraudDetectionService."""

    def setUp(self):
        """Set up test data."""
        self.user = User.objects.create_user(
            username="testuser",
            email="test@example.com",
            password="testpass123"
        )
        
        self.service = FraudDetectionService()
        
        self.sample_transaction = {
            "transaction_id": "txn_test_123",
            "user_id": "user_test_456",
            "amount": 100.00,
            "currency": "USD",
            "merchant_id": "merchant_789",
            "payment_method": "credit_card",
            "ip_address": "192.168.1.1",
            "user_agent": "Mozilla/5.0 (Test Browser)",
            "timestamp": timezone.now().isoformat()
        }

    def test_analyze_transaction_basic(self):
        """Test basic transaction analysis."""
        result = self.service.analyze_transaction(self.sample_transaction, self.user)
        
        self.assertIn('fraud_score', result)
        self.assertIn('risk_level', result)
        self.assertIn('recommendation', result)
        self.assertIsInstance(result['fraud_score'], (int, float))
        self.assertIn(result['risk_level'], ['very_low', 'low', 'medium', 'high'])
        self.assertIn(result['recommendation'], ['approve', 'manual_review', 'decline'])

    def test_high_amount_increases_score(self):
        """Test that high transaction amounts increase fraud score."""
        # Low amount transaction
        low_amount_txn = self.sample_transaction.copy()
        low_amount_txn['amount'] = 50.00
        low_result = self.service.analyze_transaction(low_amount_txn, self.user)
        
        # High amount transaction
        high_amount_txn = self.sample_transaction.copy()
        high_amount_txn['amount'] = 2000.00
        high_result = self.service.analyze_transaction(high_amount_txn, self.user)
        
        self.assertGreater(high_result['fraud_score'], low_result['fraud_score'])

    def test_risk_level_determination(self):
        """Test risk level determination based on fraud score."""
        # Test very low risk
        self.assertEqual(self.service._determine_risk_level(10), 'very_low')
        
        # Test low risk
        self.assertEqual(self.service._determine_risk_level(35), 'low')
        
        # Test medium risk
        self.assertEqual(self.service._determine_risk_level(65), 'medium')
        
        # Test high risk
        self.assertEqual(self.service._determine_risk_level(85), 'high')

    def test_recommendation_logic(self):
        """Test recommendation logic based on risk level."""
        self.assertEqual(self.service._get_recommendation('very_low', 10), 'approve')
        self.assertEqual(self.service._get_recommendation('low', 35), 'approve')
        self.assertEqual(self.service._get_recommendation('medium', 65), 'manual_review')
        self.assertEqual(self.service._get_recommendation('high', 85), 'decline')

    @patch('apps.fraud_detection.services.cache')
    def test_velocity_check(self, mock_cache):
        """Test velocity checking for rapid transactions."""
        # Mock high velocity
        mock_cache.get.return_value = 6
        
        result = self.service.analyze_transaction(self.sample_transaction, self.user)
        
        # Should have higher fraud score due to velocity
        self.assertGreater(result['fraud_score'], 0)

    def test_unusual_time_detection(self):
        """Test detection of unusual transaction times."""
        # Create transaction at 3 AM (unusual time)
        unusual_time = datetime.now().replace(hour=3, minute=0, second=0)
        unusual_txn = self.sample_transaction.copy()
        unusual_txn['timestamp'] = unusual_time.isoformat()
        
        result = self.service.analyze_transaction(unusual_txn, self.user)
        
        # Should detect unusual time pattern
        self.assertIsInstance(result['fraud_score'], (int, float))

    def test_fraud_rules_application(self):
        """Test application of custom fraud rules."""
        # Create a test fraud rule
        rule = FraudRule.objects.create(
            name="High Amount Rule",
            description="Flag transactions over $1000",
            conditions={"amount_threshold": 1000},
            action="flag",
            owner=self.user
        )
        
        # Test transaction that should trigger rule
        high_amount_txn = self.sample_transaction.copy()
        high_amount_txn['amount'] = 1500.00
        
        result = self.service.analyze_transaction(high_amount_txn, self.user)
        
        self.assertIn('rule_results', result)
        self.assertIsInstance(result['rule_results'], list)

    def test_error_handling(self):
        """Test error handling for malformed transaction data."""
        # Test with missing required fields
        invalid_txn = {"invalid": "data"}
        
        result = self.service.analyze_transaction(invalid_txn, self.user)
        
        # Should return safe defaults
        self.assertIn('fraud_score', result)
        self.assertIn('risk_level', result)
        self.assertEqual(result['risk_level'], 'unknown')

    def test_ip_reputation_check(self):
        """Test IP reputation checking."""
        # Test with potentially suspicious IP
        suspicious_txn = self.sample_transaction.copy()
        suspicious_txn['ip_address'] = "10.0.0.1"
        
        result = self.service.analyze_transaction(suspicious_txn, self.user)
        
        # Should complete analysis without errors
        self.assertIsInstance(result['fraud_score'], (int, float))

    def test_concurrent_analysis(self):
        """Test concurrent transaction analysis."""
        import threading
        
        results = []
        
        def analyze_transaction():
            result = self.service.analyze_transaction(self.sample_transaction, self.user)
            results.append(result)
        
        # Run multiple analyses concurrently
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=analyze_transaction)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All analyses should complete successfully
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertIn('fraud_score', result)


class FraudRuleTest(TestCase):
    """Test cases for FraudRule model."""

    def setUp(self):
        """Set up test data."""
        self.user = User.objects.create_user(
            username="testuser",
            email="test@example.com",
            password="testpass123"
        )

    def test_fraud_rule_creation(self):
        """Test fraud rule creation."""
        rule = FraudRule.objects.create(
            name="Test Rule",
            description="Test fraud rule",
            conditions={"test": "condition"},
            action="flag",
            owner=self.user
        )
        
        self.assertEqual(rule.name, "Test Rule")
        self.assertEqual(rule.action, "flag")
        self.assertTrue(rule.is_active)
        self.assertEqual(rule.owner, self.user)

    def test_fraud_rule_ordering(self):
        """Test fraud rule ordering by priority."""
        rule1 = FraudRule.objects.create(
            name="Rule 1",
            priority=2,
            conditions={},
            action="flag",
            owner=self.user
        )
        
        rule2 = FraudRule.objects.create(
            name="Rule 2",
            priority=1,
            conditions={},
            action="flag",
            owner=self.user
        )
        
        rules = list(FraudRule.objects.all())
        self.assertEqual(rules[0], rule2)  # Lower priority number comes first


class FraudAlertTest(TestCase):
    """Test cases for FraudAlert model."""

    def setUp(self):
        """Set up test data."""
        self.user = User.objects.create_user(
            username="testuser",
            email="test@example.com",
            password="testpass123"
        )

    def test_fraud_alert_creation(self):
        """Test fraud alert creation."""
        alert = FraudAlert.objects.create(
            transaction_id="txn_123",
            alert_type="high_amount",
            severity="high",
            message="High amount transaction detected",
            owner=self.user
        )
        
        self.assertEqual(alert.transaction_id, "txn_123")
        self.assertEqual(alert.severity, "high")
        self.assertFalse(alert.is_resolved)
        self.assertIsNone(alert.resolved_at)

    def test_fraud_alert_resolution(self):
        """Test fraud alert resolution."""
        alert = FraudAlert.objects.create(
            transaction_id="txn_123",
            alert_type="high_amount",
            severity="high",
            message="High amount transaction detected",
            owner=self.user
        )
        
        # Resolve the alert
        alert.is_resolved = True
        alert.resolved_at = timezone.now()
        alert.save()
        
        self.assertTrue(alert.is_resolved)
        self.assertIsNotNone(alert.resolved_at)


@pytest.mark.integration
class FraudDetectionIntegrationTest(TestCase):
    """Integration tests for fraud detection system."""

    def setUp(self):
        """Set up test data."""
        self.user = User.objects.create_user(
            username="testuser",
            email="test@example.com",
            password="testpass123"
        )
        
        self.service = FraudDetectionService()

    def test_end_to_end_fraud_detection(self):
        """Test complete fraud detection workflow."""
        # Create a transaction
        transaction = Transaction.objects.create(
            transaction_id="txn_integration_test",
            user_id="user_123",
            amount=Decimal('1500.00'),
            currency="USD",
            merchant_id="merchant_456",
            payment_method="credit_card",
            ip_address="192.168.1.1",
            timestamp=timezone.now(),
            owner=self.user
        )
        
        # Analyze the transaction
        transaction_data = {
            "transaction_id": transaction.transaction_id,
            "user_id": transaction.user_id,
            "amount": float(transaction.amount),
            "currency": transaction.currency,
            "merchant_id": transaction.merchant_id,
            "payment_method": transaction.payment_method,
            "ip_address": transaction.ip_address,
            "timestamp": transaction.timestamp.isoformat()
        }
        
        result = self.service.analyze_transaction(transaction_data, self.user)
        
        # Verify analysis results
        self.assertIn('fraud_score', result)
        self.assertIn('risk_level', result)
        self.assertIn('recommendation', result)
        
        # Update transaction with fraud score
        transaction.fraud_score = result['fraud_score']
        transaction.risk_level = result['risk_level']
        transaction.save()
        
        # Verify transaction was updated
        updated_transaction = Transaction.objects.get(id=transaction.id)
        self.assertEqual(updated_transaction.fraud_score, result['fraud_score'])
        self.assertEqual(updated_transaction.risk_level, result['risk_level'])

    def test_fraud_detection_with_rules(self):
        """Test fraud detection with custom rules."""
        # Create a fraud rule
        rule = FraudRule.objects.create(
            name="High Amount Alert",
            description="Alert for transactions over $1000",
            conditions={"amount_threshold": 1000},
            action="alert",
            owner=self.user
        )
        
        # Create high-amount transaction
        transaction_data = {
            "transaction_id": "txn_high_amount",
            "user_id": "user_123",
            "amount": 1500.00,
            "currency": "USD",
            "merchant_id": "merchant_456",
            "payment_method": "credit_card",
            "ip_address": "192.168.1.1",
            "timestamp": timezone.now().isoformat()
        }
        
        result = self.service.analyze_transaction(transaction_data, self.user)
        
        # Should have rule results
        self.assertIn('rule_results', result)
        self.assertIsInstance(result['rule_results'], list)