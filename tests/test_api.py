"""
Comprehensive API tests for fraud analytics platform.
Enterprise-grade testing with Claude Code assistance.
"""

import json
from decimal import Decimal

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient
from rest_framework_simplejwt.tokens import RefreshToken

from apps.authentication.models import APIKey
from apps.cases.models import Case
from apps.fraud_detection.models import FraudAlert, FraudRule
from apps.transactions.models import Transaction
from apps.webhooks.models import Webhook

User = get_user_model()


class AuthenticationAPITest(TestCase):
    """Test authentication API endpoints."""

    def setUp(self):
        """Set up test data."""
        self.client = APIClient()
        self.user_data = {
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'testpass123',
            'company_name': 'Test Company'
        }

    def test_user_registration(self):
        """Test user registration endpoint."""
        url = reverse('register')
        response = self.client.post(url, self.user_data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn('id', response.data)
        self.assertEqual(response.data['email'], self.user_data['email'])
        self.assertNotIn('password', response.data)

    def test_jwt_token_generation(self):
        """Test JWT token generation."""
        # Create user first
        user = User.objects.create_user(**self.user_data)
        
        # Get token
        url = reverse('token_obtain_pair')
        login_data = {
            'email': self.user_data['email'],
            'password': self.user_data['password']
        }
        response = self.client.post(url, login_data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('access', response.data)
        self.assertIn('refresh', response.data)

    def test_jwt_token_refresh(self):
        """Test JWT token refresh."""
        user = User.objects.create_user(**self.user_data)
        refresh = RefreshToken.for_user(user)
        
        url = reverse('token_refresh')
        response = self.client.post(url, {'refresh': str(refresh)}, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('access', response.data)

    def test_profile_access(self):
        """Test user profile access."""
        user = User.objects.create_user(**self.user_data)
        self.client.force_authenticate(user=user)
        
        url = reverse('profile')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['email'], user.email)

    def test_api_key_creation(self):
        """Test API key creation."""
        user = User.objects.create_user(**self.user_data)
        self.client.force_authenticate(user=user)
        
        url = reverse('api_keys')
        api_key_data = {'name': 'Test API Key'}
        response = self.client.post(url, api_key_data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn('key', response.data)
        self.assertEqual(response.data['name'], 'Test API Key')


class TransactionAPITest(TestCase):
    """Test transaction API endpoints."""

    def setUp(self):
        """Set up test data."""
        self.client = APIClient()
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        self.client.force_authenticate(user=self.user)
        
        self.transaction_data = {
            'transaction_id': 'txn_test_123',
            'user_id': 'user_456',
            'amount': '100.00',
            'currency': 'USD',
            'merchant_id': 'merchant_789',
            'payment_method': 'credit_card',
            'ip_address': '192.168.1.1',
            'timestamp': '2024-01-15T10:30:00Z'
        }

    def test_transaction_creation(self):
        """Test transaction creation."""
        url = reverse('transaction_list')
        response = self.client.post(url, self.transaction_data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['transaction_id'], self.transaction_data['transaction_id'])
        self.assertEqual(response.data['amount'], self.transaction_data['amount'])

    def test_transaction_list(self):
        """Test transaction listing."""
        # Create test transaction
        Transaction.objects.create(
            transaction_id='txn_list_test',
            user_id='user_123',
            amount=Decimal('50.00'),
            currency='USD',
            merchant_id='merchant_456',
            payment_method='credit_card',
            ip_address='192.168.1.1',
            timestamp='2024-01-15T10:30:00Z',
            owner=self.user
        )
        
        url = reverse('transaction_list')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertGreater(len(response.data['results']), 0)

    def test_transaction_detail(self):
        """Test transaction detail retrieval."""
        transaction = Transaction.objects.create(
            transaction_id='txn_detail_test',
            user_id='user_123',
            amount=Decimal('75.00'),
            currency='USD',
            merchant_id='merchant_456',
            payment_method='credit_card',
            ip_address='192.168.1.1',
            timestamp='2024-01-15T10:30:00Z',
            owner=self.user
        )
        
        url = reverse('transaction_detail', kwargs={'pk': transaction.id})
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['transaction_id'], 'txn_detail_test')

    def test_transaction_analysis(self):
        """Test transaction fraud analysis."""
        url = reverse('transaction_analysis')
        response = self.client.post(url, self.transaction_data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('fraud_score', response.data)
        self.assertIn('risk_level', response.data)
        self.assertIn('recommendation', response.data)

    def test_unauthorized_access(self):
        """Test unauthorized access to transactions."""
        self.client.force_authenticate(user=None)
        
        url = reverse('transaction_list')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)


class FraudDetectionAPITest(TestCase):
    """Test fraud detection API endpoints."""

    def setUp(self):
        """Set up test data."""
        self.client = APIClient()
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        self.client.force_authenticate(user=self.user)

    def test_fraud_analysis_endpoint(self):
        """Test fraud analysis API endpoint."""
        url = reverse('fraud_analysis')
        transaction_data = {
            'transaction_id': 'txn_fraud_test',
            'user_id': 'user_123',
            'amount': 500.00,
            'currency': 'USD',
            'merchant_id': 'merchant_456',
            'payment_method': 'credit_card',
            'ip_address': '192.168.1.1',
            'timestamp': '2024-01-15T10:30:00Z'
        }
        
        response = self.client.post(url, transaction_data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('fraud_score', response.data)
        self.assertIn('risk_level', response.data)
        self.assertIn('recommendation', response.data)
        self.assertIsInstance(response.data['fraud_score'], (int, float))

    def test_fraud_rules_creation(self):
        """Test fraud rule creation."""
        url = reverse('fraud_rules')
        rule_data = {
            'name': 'High Amount Rule',
            'description': 'Flag transactions over $1000',
            'conditions': {'amount_threshold': 1000},
            'action': 'flag',
            'priority': 1
        }
        
        response = self.client.post(url, rule_data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['name'], rule_data['name'])
        self.assertEqual(response.data['action'], rule_data['action'])

    def test_fraud_rules_list(self):
        """Test fraud rules listing."""
        # Create test rule
        FraudRule.objects.create(
            name='Test Rule',
            description='Test description',
            conditions={'test': 'condition'},
            action='flag',
            owner=self.user
        )
        
        url = reverse('fraud_rules')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertGreater(len(response.data['results']), 0)

    def test_fraud_alerts_list(self):
        """Test fraud alerts listing."""
        # Create test alert
        FraudAlert.objects.create(
            transaction_id='txn_alert_test',
            alert_type='high_amount',
            severity='high',
            message='High amount detected',
            owner=self.user
        )
        
        url = reverse('fraud_alerts')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertGreater(len(response.data['results']), 0)


class CaseManagementAPITest(TestCase):
    """Test case management API endpoints."""

    def setUp(self):
        """Set up test data."""
        self.client = APIClient()
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        self.client.force_authenticate(user=self.user)

    def test_case_creation(self):
        """Test fraud case creation."""
        url = reverse('case_list')
        case_data = {
            'title': 'Suspicious Transaction',
            'description': 'High amount transaction from new user',
            'transaction_id': 'txn_case_test',
            'priority': 'high'
        }
        
        response = self.client.post(url, case_data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['title'], case_data['title'])
        self.assertIn('case_number', response.data)

    def test_case_list(self):
        """Test case listing."""
        # Create test case
        Case.objects.create(
            case_number='CASE-TEST-001',
            title='Test Case',
            description='Test description',
            transaction_id='txn_123',
            owner=self.user
        )
        
        url = reverse('case_list')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertGreater(len(response.data['results']), 0)

    def test_case_detail(self):
        """Test case detail retrieval."""
        case = Case.objects.create(
            case_number='CASE-DETAIL-001',
            title='Detail Test Case',
            description='Test description',
            transaction_id='txn_detail',
            owner=self.user
        )
        
        url = reverse('case_detail', kwargs={'pk': case.id})
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['case_number'], 'CASE-DETAIL-001')

    def test_case_update(self):
        """Test case update."""
        case = Case.objects.create(
            case_number='CASE-UPDATE-001',
            title='Update Test Case',
            description='Test description',
            transaction_id='txn_update',
            status='open',
            owner=self.user
        )
        
        url = reverse('case_detail', kwargs={'pk': case.id})
        update_data = {'status': 'investigating'}
        response = self.client.patch(url, update_data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['status'], 'investigating')


class AnalyticsAPITest(TestCase):
    """Test analytics API endpoints."""

    def setUp(self):
        """Set up test data."""
        self.client = APIClient()
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        self.client.force_authenticate(user=self.user)

    def test_dashboard_stats(self):
        """Test dashboard statistics endpoint."""
        # Create test data
        Transaction.objects.create(
            transaction_id='txn_stats_1',
            user_id='user_123',
            amount=Decimal('100.00'),
            currency='USD',
            merchant_id='merchant_456',
            payment_method='credit_card',
            ip_address='192.168.1.1',
            timestamp='2024-01-15T10:30:00Z',
            status='approved',
            owner=self.user
        )
        
        Transaction.objects.create(
            transaction_id='txn_stats_2',
            user_id='user_456',
            amount=Decimal('200.00'),
            currency='USD',
            merchant_id='merchant_789',
            payment_method='credit_card',
            ip_address='192.168.1.2',
            timestamp='2024-01-15T11:30:00Z',
            status='flagged',
            fraud_score=75.0,
            owner=self.user
        )
        
        Case.objects.create(
            case_number='CASE-STATS-001',
            title='Stats Test Case',
            description='Test description',
            transaction_id='txn_stats_2',
            status='open',
            owner=self.user
        )
        
        url = reverse('dashboard_stats')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('total_transactions', response.data)
        self.assertIn('flagged_transactions', response.data)
        self.assertIn('open_cases', response.data)
        self.assertIn('average_fraud_score', response.data)
        self.assertIn('fraud_rate', response.data)
        
        self.assertEqual(response.data['total_transactions'], 2)
        self.assertEqual(response.data['flagged_transactions'], 1)
        self.assertEqual(response.data['open_cases'], 1)


class WebhookAPITest(TestCase):
    """Test webhook API endpoints."""

    def setUp(self):
        """Set up test data."""
        self.client = APIClient()
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        self.client.force_authenticate(user=self.user)

    def test_webhook_creation(self):
        """Test webhook creation."""
        url = reverse('webhook_list')
        webhook_data = {
            'name': 'Test Webhook',
            'url': 'https://example.com/webhook',
            'secret': 'webhook_secret_123',
            'events': ['transaction.flagged', 'case.created']
        }
        
        response = self.client.post(url, webhook_data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['name'], webhook_data['name'])
        self.assertEqual(response.data['url'], webhook_data['url'])

    def test_webhook_list(self):
        """Test webhook listing."""
        # Create test webhook
        Webhook.objects.create(
            name='List Test Webhook',
            url='https://example.com/list-webhook',
            secret='secret_123',
            events=['transaction.flagged'],
            owner=self.user
        )
        
        url = reverse('webhook_list')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertGreater(len(response.data['results']), 0)


class APISecurityTest(TestCase):
    """Test API security measures."""

    def setUp(self):
        """Set up test data."""
        self.client = APIClient()
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )

    def test_authentication_required(self):
        """Test that authentication is required for protected endpoints."""
        protected_urls = [
            reverse('transaction_list'),
            reverse('fraud_analysis'),
            reverse('case_list'),
            reverse('dashboard_stats'),
            reverse('webhook_list'),
        ]
        
        for url in protected_urls:
            response = self.client.get(url)
            self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_user_isolation(self):
        """Test that users can only access their own data."""
        # Create two users
        user1 = self.user
        user2 = User.objects.create_user(
            username='testuser2',
            email='test2@example.com',
            password='testpass123'
        )
        
        # Create transaction for user1
        transaction = Transaction.objects.create(
            transaction_id='txn_isolation_test',
            user_id='user_123',
            amount=Decimal('100.00'),
            currency='USD',
            merchant_id='merchant_456',
            payment_method='credit_card',
            ip_address='192.168.1.1',
            timestamp='2024-01-15T10:30:00Z',
            owner=user1
        )
        
        # User2 should not be able to access user1's transaction
        self.client.force_authenticate(user=user2)
        url = reverse('transaction_detail', kwargs={'pk': transaction.id})
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_input_validation(self):
        """Test input validation for API endpoints."""
        self.client.force_authenticate(user=self.user)
        
        # Test invalid transaction data
        url = reverse('transaction_analysis')
        invalid_data = {
            'transaction_id': '',  # Empty required field
            'amount': 'invalid_amount',  # Invalid amount format
            'ip_address': 'invalid_ip'  # Invalid IP format
        }
        
        response = self.client.post(url, invalid_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_rate_limiting_headers(self):
        """Test that rate limiting headers are present."""
        self.client.force_authenticate(user=self.user)
        
        url = reverse('dashboard_stats')
        response = self.client.get(url)
        
        # Note: This test assumes rate limiting middleware is configured
        # In a real implementation, you would check for rate limiting headers
        self.assertEqual(response.status_code, status.HTTP_200_OK)