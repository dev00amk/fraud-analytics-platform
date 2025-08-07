from datetime import timedelta
from decimal import Decimal
from django.test import TestCase
from django.contrib.auth import get_user_model
from django.utils import timezone
from apps.transactions.models import Transaction
from .models import (
    Alert, AlertRule, NotificationDelivery, 
    EscalationRule, NotificationTemplate, EscalationTask
)

User = get_user_model()


class AlertModelTests(TestCase):
    """Test cases for Alert model."""
    
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        
        self.transaction = Transaction.objects.create(
            transaction_id='txn_123',
            user_id='user_456',
            amount=Decimal('100.00'),
            currency='USD',
            merchant_id='merchant_789',
            payment_method='credit_card',
            ip_address='192.168.1.1',
            timestamp=timezone.now(),
            owner=self.user
        )
    
    def test_alert_creation(self):
        """Test creating an alert."""
        alert = Alert.objects.create(
            alert_type='high_risk_transaction',
            severity='high',
            transaction=self.transaction,
            fraud_score=0.85,
            risk_factors={'velocity': 'high', 'amount': 'unusual'},
            title='High Risk Transaction Detected',
            message='Transaction flagged for manual review',
            context_data={'ip_country': 'Unknown'},
            owner=self.user
        )
        
        self.assertEqual(alert.alert_type, 'high_risk_transaction')
        self.assertEqual(alert.severity, 'high')
        self.assertEqual(alert.status, 'pending')
        self.assertEqual(alert.fraud_score, 0.85)
        self.assertEqual(alert.owner, self.user)
        self.assertEqual(str(alert), f"Alert: high_risk_transaction - {self.transaction.transaction_id}")
    
    def test_alert_acknowledgment(self):
        """Test alert acknowledgment."""
        alert = Alert.objects.create(
            alert_type='suspicious_activity',
            severity='medium',
            transaction=self.transaction,
            fraud_score=0.65,
            title='Suspicious Activity',
            message='Unusual transaction pattern detected',
            owner=self.user
        )
        
        # Acknowledge the alert
        alert.status = 'acknowledged'
        alert.acknowledged_at = timezone.now()
        alert.acknowledged_by = self.user
        alert.save()
        
        self.assertEqual(alert.status, 'acknowledged')
        self.assertIsNotNone(alert.acknowledged_at)
        self.assertEqual(alert.acknowledged_by, self.user)


class AlertRuleModelTests(TestCase):
    """Test cases for AlertRule model."""
    
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
    
    def test_alert_rule_creation(self):
        """Test creating an alert rule."""
        rule = AlertRule.objects.create(
            name='High Amount Rule',
            description='Alert for transactions over $1000',
            conditions={
                'amount_threshold': {
                    'type': 'threshold',
                    'field': 'amount',
                    'operator': '>',
                    'threshold': 1000
                }
            },
            fraud_score_threshold=0.7,
            amount_threshold=Decimal('1000.00'),
            alert_type='high_amount',
            severity='high',
            action='alert',
            notification_channels=['email', 'slack'],
            consolidation_window=timedelta(minutes=15),
            owner=self.user
        )
        
        self.assertEqual(rule.name, 'High Amount Rule')
        self.assertEqual(rule.severity, 'high')
        self.assertEqual(rule.action, 'alert')
        self.assertTrue(rule.is_active)
        self.assertEqual(rule.priority, 1)
        self.assertEqual(str(rule), "Rule: High Amount Rule")
    
    def test_alert_rule_conditions(self):
        """Test alert rule conditions structure."""
        rule = AlertRule.objects.create(
            name='Complex Rule',
            conditions={
                'fraud_score': {
                    'type': 'threshold',
                    'field': 'fraud_score',
                    'operator': '>=',
                    'threshold': 0.8
                },
                'velocity': {
                    'type': 'threshold',
                    'field': 'feature_velocity_1h',
                    'operator': '>',
                    'threshold': 5
                },
                '_meta': {
                    'required_conditions': 2
                }
            },
            alert_type='complex_fraud',
            severity='critical',
            action='escalate',
            owner=self.user
        )
        
        self.assertIn('fraud_score', rule.conditions)
        self.assertIn('velocity', rule.conditions)
        self.assertEqual(rule.conditions['_meta']['required_conditions'], 2)


class NotificationDeliveryModelTests(TestCase):
    """Test cases for NotificationDelivery model."""
    
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        
        self.transaction = Transaction.objects.create(
            transaction_id='txn_123',
            user_id='user_456',
            amount=Decimal('100.00'),
            currency='USD',
            merchant_id='merchant_789',
            payment_method='credit_card',
            ip_address='192.168.1.1',
            timestamp=timezone.now(),
            owner=self.user
        )
        
        self.alert = Alert.objects.create(
            alert_type='test_alert',
            severity='medium',
            transaction=self.transaction,
            fraud_score=0.6,
            title='Test Alert',
            message='Test message',
            owner=self.user
        )
    
    def test_notification_delivery_creation(self):
        """Test creating a notification delivery record."""
        delivery = NotificationDelivery.objects.create(
            alert=self.alert,
            channel_type='email',
            recipient='test@example.com',
            status='pending'
        )
        
        self.assertEqual(delivery.alert, self.alert)
        self.assertEqual(delivery.channel_type, 'email')
        self.assertEqual(delivery.recipient, 'test@example.com')
        self.assertEqual(delivery.status, 'pending')
        self.assertEqual(delivery.attempts, 0)
        self.assertEqual(str(delivery), "Delivery: email to test@example.com")
    
    def test_notification_delivery_retry(self):
        """Test notification delivery retry logic."""
        delivery = NotificationDelivery.objects.create(
            alert=self.alert,
            channel_type='sms',
            recipient='+1234567890',
            status='failed',
            attempts=1,
            error_message='Network timeout',
            retry_after=timezone.now() + timedelta(minutes=5)
        )
        
        self.assertEqual(delivery.status, 'failed')
        self.assertEqual(delivery.attempts, 1)
        self.assertEqual(delivery.error_message, 'Network timeout')
        self.assertIsNotNone(delivery.retry_after)


class EscalationRuleModelTests(TestCase):
    """Test cases for EscalationRule model."""
    
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
    
    def test_escalation_rule_creation(self):
        """Test creating an escalation rule."""
        escalation_rule = EscalationRule.objects.create(
            name='Critical Alert Escalation',
            alert_severity='critical',
            timeout_minutes=15,
            business_hours_only=False,
            escalation_levels=[
                {
                    'level': 1,
                    'timeout_minutes': 15,
                    'recipients': ['analyst@example.com'],
                    'channels': ['email', 'slack']
                },
                {
                    'level': 2,
                    'timeout_minutes': 30,
                    'recipients': ['manager@example.com'],
                    'channels': ['email', 'sms']
                }
            ],
            owner=self.user
        )
        
        self.assertEqual(escalation_rule.name, 'Critical Alert Escalation')
        self.assertEqual(escalation_rule.alert_severity, 'critical')
        self.assertEqual(escalation_rule.timeout_minutes, 15)
        self.assertFalse(escalation_rule.business_hours_only)
        self.assertEqual(len(escalation_rule.escalation_levels), 2)
        self.assertTrue(escalation_rule.is_active)


class NotificationTemplateModelTests(TestCase):
    """Test cases for NotificationTemplate model."""
    
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
    
    def test_notification_template_creation(self):
        """Test creating a notification template."""
        template = NotificationTemplate.objects.create(
            name='High Risk Email Template',
            channel_type='email',
            alert_type='high_risk_transaction',
            subject_template='FRAUD ALERT: High Risk Transaction - {transaction_id}',
            body_template='''
            A high risk transaction has been detected:
            
            Transaction ID: {transaction_id}
            Amount: {amount} {currency}
            Fraud Score: {fraud_score}
            Risk Level: {risk_level}
            
            Please review immediately.
            ''',
            variables={
                'transaction_id': 'string',
                'amount': 'decimal',
                'currency': 'string',
                'fraud_score': 'float',
                'risk_level': 'string'
            },
            formatting_options={
                'html_enabled': True,
                'include_logo': True
            },
            is_default=True,
            owner=self.user
        )
        
        self.assertEqual(template.name, 'High Risk Email Template')
        self.assertEqual(template.channel_type, 'email')
        self.assertEqual(template.alert_type, 'high_risk_transaction')
        self.assertTrue(template.is_default)
        self.assertIn('transaction_id', template.variables)
        self.assertEqual(str(template), "Template: High Risk Email Template (email)")


class EscalationTaskModelTests(TestCase):
    """Test cases for EscalationTask model."""
    
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        
        self.transaction = Transaction.objects.create(
            transaction_id='txn_123',
            user_id='user_456',
            amount=Decimal('100.00'),
            currency='USD',
            merchant_id='merchant_789',
            payment_method='credit_card',
            ip_address='192.168.1.1',
            timestamp=timezone.now(),
            owner=self.user
        )
        
        self.alert = Alert.objects.create(
            alert_type='critical_fraud',
            severity='critical',
            transaction=self.transaction,
            fraud_score=0.95,
            title='Critical Fraud Alert',
            message='Immediate attention required',
            owner=self.user
        )
        
        self.escalation_rule = EscalationRule.objects.create(
            name='Critical Escalation',
            alert_severity='critical',
            timeout_minutes=10,
            escalation_levels=[
                {'level': 1, 'timeout_minutes': 10, 'recipients': ['manager@example.com']}
            ],
            owner=self.user
        )
    
    def test_escalation_task_creation(self):
        """Test creating an escalation task."""
        task = EscalationTask.objects.create(
            alert=self.alert,
            escalation_rule=self.escalation_rule,
            escalation_level=1,
            scheduled_at=timezone.now() + timedelta(minutes=10)
        )
        
        self.assertEqual(task.alert, self.alert)
        self.assertEqual(task.escalation_rule, self.escalation_rule)
        self.assertEqual(task.escalation_level, 1)
        self.assertEqual(task.status, 'scheduled')
        self.assertIsNotNone(task.scheduled_at)
        self.assertEqual(str(task), f"Escalation Task: {self.alert.alert_type} - Level 1")
    
    def test_escalation_task_execution(self):
        """Test escalation task execution."""
        task = EscalationTask.objects.create(
            alert=self.alert,
            escalation_rule=self.escalation_rule,
            escalation_level=1,
            scheduled_at=timezone.now() + timedelta(minutes=10)
        )
        
        # Execute the task
        task.status = 'executed'
        task.executed_at = timezone.now()
        task.save()
        
        self.assertEqual(task.status, 'executed')
        self.assertIsNotNone(task.executed_at)
    
    def test_escalation_task_cancellation(self):
        """Test escalation task cancellation."""
        task = EscalationTask.objects.create(
            alert=self.alert,
            escalation_rule=self.escalation_rule,
            escalation_level=1,
            scheduled_at=timezone.now() + timedelta(minutes=10)
        )
        
        # Cancel the task
        task.status = 'cancelled'
        task.cancelled_at = timezone.now()
        task.save()
        
        self.assertEqual(task.status, 'cancelled')
        self.assertIsNotNone(task.cancelled_at)