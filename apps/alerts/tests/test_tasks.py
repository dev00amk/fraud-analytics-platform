"""
Tests for alert processing tasks and queue functionality.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from django.test import TestCase, override_settings
from django.contrib.auth import get_user_model
from django.utils import timezone
from django.core.cache import cache

from apps.transactions.models import Transaction
from apps.alerts.models import Alert, AlertRule, NotificationDelivery, EscalationRule, EscalationTask
from apps.alerts.tasks import (
    process_critical_alert,
    process_high_priority_alert,
    process_medium_priority_alert,
    process_low_priority_alert,
    send_notification_task,
    schedule_escalation_task,
    execute_escalation_task,
    cleanup_failed_deliveries,
    health_check_task,
    _check_rate_limit,
    _process_single_alert
)

User = get_user_model()


class AlertProcessingTasksTest(TestCase):
    """Test alert processing tasks with priority queues."""
    
    def setUp(self):
        """Set up test data."""
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        
        self.transaction = Transaction.objects.create(
            transaction_id='test_txn_001',
            user_id=self.user.id,
            amount=100.00,
            currency='USD',
            merchant='Test Merchant',
            status='completed'
        )
        
        self.alert_rule = AlertRule.objects.create(
            name='Test Rule',
            alert_type='high_risk_transaction',
            severity='high',
            action='alert',
            fraud_score_threshold=80.0,
            notification_channels=['email'],
            owner=self.user
        )
        
        self.alert = Alert.objects.create(
            alert_type='high_risk_transaction',
            severity='high',
            transaction=self.transaction,
            fraud_score=85.0,
            title='High Risk Transaction Detected',
            message='Transaction flagged for manual review',
            rule_triggered=self.alert_rule,
            owner=self.user
        )
    
    def tearDown(self):
        """Clean up after tests."""
        cache.clear()
    
    @patch('apps.alerts.tasks._process_single_alert')
    def test_process_critical_alert_success(self, mock_process):
        """Test successful processing of critical alert."""
        mock_process.return_value = {
            'success': True,
            'notifications_sent': 2,
            'failed_notifications': 0
        }
        
        # Create critical alert
        critical_alert = Alert.objects.create(
            alert_type='critical_fraud',
            severity='critical',
            transaction=self.transaction,
            fraud_score=95.0,
            title='Critical Fraud Detected',
            message='Immediate action required',
            owner=self.user
        )
        
        # Process the alert
        result = process_critical_alert.apply(args=[str(critical_alert.id)])
        
        self.assertTrue(result.successful())
        result_data = result.result
        self.assertEqual(result_data['status'], 'processed')
        self.assertEqual(result_data['notifications_sent'], 2)
        self.assertTrue(result_data['success'])
        
        # Verify alert status updated
        critical_alert.refresh_from_db()
        self.assertEqual(critical_alert.status, 'sent')
    
    @patch('apps.alerts.tasks._process_single_alert')
    def test_process_alert_with_rate_limiting(self, mock_process):
        """Test alert processing with rate limiting."""
        # Set up rate limiting
        rate_limit_key = f"alert_rate_limit:{self.alert.alert_type}:{self.alert.owner.id}"
        cache.set(rate_limit_key, 10, 60)  # Max limit reached
        
        result = process_high_priority_alert.apply(args=[str(self.alert.id)])
        
        self.assertTrue(result.successful())
        result_data = result.result
        self.assertEqual(result_data['status'], 'rate_limited')
        
        # Verify process_single_alert was not called
        mock_process.assert_not_called()
    
    def test_process_nonexistent_alert(self):
        """Test processing of non-existent alert."""
        fake_id = '00000000-0000-0000-0000-000000000000'
        
        result = process_high_priority_alert.apply(args=[fake_id])
        
        self.assertTrue(result.successful())
        result_data = result.result
        self.assertEqual(result_data['status'], 'not_found')
        self.assertIn('error', result_data)
    
    @patch('apps.alerts.tasks._process_single_alert')
    def test_process_alert_with_exception(self, mock_process):
        """Test alert processing with exception handling."""
        mock_process.side_effect = Exception("Processing error")
        
        # This should trigger retry mechanism
        with self.assertRaises(Exception):
            process_high_priority_alert.apply(args=[str(self.alert.id)])
    
    def test_rate_limit_check(self):
        """Test rate limiting functionality."""
        # First call should pass
        self.assertTrue(_check_rate_limit(self.alert))
        
        # Simulate multiple calls to reach limit
        rate_limit_key = f"alert_rate_limit:{self.alert.alert_type}:{self.alert.owner.id}"
        cache.set(rate_limit_key, 10, 60)  # Set to max limit
        
        # Next call should be rate limited
        self.assertFalse(_check_rate_limit(self.alert))
    
    @patch('apps.alerts.channels.router.AlertRouter.route_alert')
    def test_process_single_alert_success(self, mock_router):
        """Test successful processing of single alert."""
        mock_router.return_value = [
            {
                'channel_type': 'email',
                'recipient': 'test@example.com',
                'priority': 'high'
            }
        ]
        
        with patch('apps.alerts.tasks.send_notification_task.apply_async') as mock_task:
            result = _process_single_alert(self.alert)
            
            self.assertTrue(result['success'])
            self.assertEqual(result['notifications_sent'], 1)
            self.assertEqual(result['failed_notifications'], 0)
            
            # Verify notification delivery was created
            delivery = NotificationDelivery.objects.filter(alert=self.alert).first()
            self.assertIsNotNone(delivery)
            self.assertEqual(delivery.channel_type, 'email')
            self.assertEqual(delivery.recipient, 'test@example.com')
            
            # Verify task was queued
            mock_task.assert_called_once()


class NotificationTasksTest(TestCase):
    """Test notification delivery tasks."""
    
    def setUp(self):
        """Set up test data."""
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        
        self.transaction = Transaction.objects.create(
            transaction_id='test_txn_001',
            user_id=self.user.id,
            amount=100.00,
            currency='USD',
            merchant='Test Merchant',
            status='completed'
        )
        
        self.alert = Alert.objects.create(
            alert_type='test_alert',
            severity='medium',
            transaction=self.transaction,
            fraud_score=75.0,
            title='Test Alert',
            message='Test message',
            owner=self.user
        )
        
        self.delivery = NotificationDelivery.objects.create(
            alert=self.alert,
            channel_type='email',
            recipient='test@example.com',
            status='pending'
        )
    
    @patch('apps.alerts.channels.factory.NotificationChannelFactory.get_channel')
    def test_send_notification_success(self, mock_get_channel):
        """Test successful notification delivery."""
        # Mock channel and result
        mock_channel = Mock()
        mock_result = Mock()
        mock_result.success = True
        mock_result.delivered = True
        mock_result.external_id = 'ext_123'
        mock_result.error_message = None
        mock_result.should_retry = False
        
        mock_channel.send_notification.return_value = mock_result
        mock_get_channel.return_value = mock_channel
        
        result = send_notification_task.apply(args=[str(self.delivery.id)])
        
        self.assertTrue(result.successful())
        result_data = result.result
        self.assertEqual(result_data['status'], 'delivered')
        self.assertTrue(result_data['success'])
        
        # Verify delivery record updated
        self.delivery.refresh_from_db()
        self.assertEqual(self.delivery.status, 'delivered')
        self.assertEqual(self.delivery.external_id, 'ext_123')
        self.assertEqual(self.delivery.attempts, 1)
        self.assertIsNotNone(self.delivery.delivered_at)
    
    @patch('apps.alerts.channels.factory.NotificationChannelFactory.get_channel')
    def test_send_notification_failure_with_retry(self, mock_get_channel):
        """Test notification delivery failure with retry."""
        # Mock channel and result
        mock_channel = Mock()
        mock_result = Mock()
        mock_result.success = False
        mock_result.delivered = False
        mock_result.external_id = None
        mock_result.error_message = 'Temporary failure'
        mock_result.should_retry = True
        
        mock_channel.send_notification.return_value = mock_result
        mock_get_channel.return_value = mock_channel
        
        with patch.object(send_notification_task, 'apply_async') as mock_retry:
            result = send_notification_task.apply(args=[str(self.delivery.id)])
            
            self.assertTrue(result.successful())
            result_data = result.result
            self.assertEqual(result_data['status'], 'retrying')
            
            # Verify delivery record updated
            self.delivery.refresh_from_db()
            self.assertEqual(self.delivery.status, 'retrying')
            self.assertEqual(self.delivery.error_message, 'Temporary failure')
            self.assertEqual(self.delivery.attempts, 1)
            self.assertIsNotNone(self.delivery.retry_after)
            
            # Verify retry was scheduled
            mock_retry.assert_called_once()
    
    @patch('apps.alerts.channels.factory.NotificationChannelFactory.get_channel')
    def test_send_notification_permanent_failure(self, mock_get_channel):
        """Test notification delivery permanent failure."""
        # Set delivery to max attempts
        self.delivery.attempts = 3
        self.delivery.save()
        
        # Mock channel and result
        mock_channel = Mock()
        mock_result = Mock()
        mock_result.success = False
        mock_result.delivered = False
        mock_result.external_id = None
        mock_result.error_message = 'Permanent failure'
        mock_result.should_retry = False
        
        mock_channel.send_notification.return_value = mock_result
        mock_get_channel.return_value = mock_channel
        
        result = send_notification_task.apply(args=[str(self.delivery.id)])
        
        self.assertTrue(result.successful())
        result_data = result.result
        self.assertEqual(result_data['status'], 'failed')
        
        # Verify delivery record updated
        self.delivery.refresh_from_db()
        self.assertEqual(self.delivery.status, 'failed')
        self.assertEqual(self.delivery.error_message, 'Permanent failure')
        self.assertEqual(self.delivery.attempts, 4)  # Incremented
    
    def test_send_notification_already_processed(self):
        """Test sending notification that's already processed."""
        self.delivery.status = 'delivered'
        self.delivery.save()
        
        result = send_notification_task.apply(args=[str(self.delivery.id)])
        
        self.assertTrue(result.successful())
        result_data = result.result
        self.assertEqual(result_data['status'], 'already_processed')
    
    def test_send_notification_not_found(self):
        """Test sending notification for non-existent delivery."""
        fake_id = '00000000-0000-0000-0000-000000000000'
        
        result = send_notification_task.apply(args=[fake_id])
        
        self.assertTrue(result.successful())
        result_data = result.result
        self.assertEqual(result_data['status'], 'not_found')


class EscalationTasksTest(TestCase):
    """Test escalation tasks."""
    
    def setUp(self):
        """Set up test data."""
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        
        self.transaction = Transaction.objects.create(
            transaction_id='test_txn_001',
            user_id=self.user.id,
            amount=100.00,
            currency='USD',
            merchant='Test Merchant',
            status='completed'
        )
        
        self.alert = Alert.objects.create(
            alert_type='test_alert',
            severity='high',
            transaction=self.transaction,
            fraud_score=85.0,
            title='Test Alert',
            message='Test message',
            owner=self.user
        )
        
        self.escalation_rule = EscalationRule.objects.create(
            name='Test Escalation',
            alert_severity='high',
            timeout_minutes=30,
            escalation_levels=[
                {
                    'recipients': [{'user_id': self.user.id}],
                    'channels': ['email']
                }
            ],
            owner=self.user
        )
    
    def test_schedule_escalation_success(self):
        """Test successful escalation scheduling."""
        result = schedule_escalation_task.apply(args=[str(self.alert.id)])
        
        self.assertTrue(result.successful())
        result_data = result.result
        self.assertEqual(result_data['status'], 'success')
        self.assertEqual(result_data['scheduled_tasks'], 1)
        
        # Verify escalation task was created
        escalation_task = EscalationTask.objects.filter(alert=self.alert).first()
        self.assertIsNotNone(escalation_task)
        self.assertEqual(escalation_task.escalation_rule, self.escalation_rule)
        self.assertEqual(escalation_task.status, 'scheduled')
    
    def test_schedule_escalation_no_rules(self):
        """Test escalation scheduling with no applicable rules."""
        # Create alert with severity that has no rules
        low_alert = Alert.objects.create(
            alert_type='test_alert',
            severity='low',
            transaction=self.transaction,
            fraud_score=25.0,
            title='Low Alert',
            message='Test message',
            owner=self.user
        )
        
        result = schedule_escalation_task.apply(args=[str(low_alert.id)])
        
        self.assertTrue(result.successful())
        result_data = result.result
        self.assertEqual(result_data['scheduled_tasks'], 0)
    
    @patch('apps.alerts.services.escalation.EscalationEngine.execute_escalation')
    def test_execute_escalation_success(self, mock_execute):
        """Test successful escalation execution."""
        # Create escalation task
        escalation_task = EscalationTask.objects.create(
            alert=self.alert,
            escalation_rule=self.escalation_rule,
            escalation_level=1,
            scheduled_at=timezone.now(),
            status='scheduled'
        )
        
        # Mock escalation result
        mock_result = Mock()
        mock_result.get.side_effect = lambda key, default=None: {
            'notifications_sent': 1,
            'schedule_next_level': False
        }.get(key, default)
        
        mock_execute.return_value = mock_result
        
        result = execute_escalation_task.apply(args=[str(escalation_task.id)])
        
        self.assertTrue(result.successful())
        result_data = result.result
        self.assertEqual(result_data['status'], 'executed')
        self.assertEqual(result_data['notifications_sent'], 1)
        
        # Verify escalation task updated
        escalation_task.refresh_from_db()
        self.assertEqual(escalation_task.status, 'executed')
        self.assertIsNotNone(escalation_task.executed_at)
    
    def test_execute_escalation_cancelled_alert(self):
        """Test escalation execution for already acknowledged alert."""
        # Set alert as acknowledged
        self.alert.status = 'acknowledged'
        self.alert.save()
        
        # Create escalation task
        escalation_task = EscalationTask.objects.create(
            alert=self.alert,
            escalation_rule=self.escalation_rule,
            escalation_level=1,
            scheduled_at=timezone.now(),
            status='scheduled'
        )
        
        result = execute_escalation_task.apply(args=[str(escalation_task.id)])
        
        self.assertTrue(result.successful())
        result_data = result.result
        self.assertEqual(result_data['status'], 'cancelled')
        self.assertIn('already acknowledged', result_data['reason'])
        
        # Verify escalation task cancelled
        escalation_task.refresh_from_db()
        self.assertEqual(escalation_task.status, 'cancelled')
        self.assertIsNotNone(escalation_task.cancelled_at)


class MaintenanceTasksTest(TestCase):
    """Test maintenance and cleanup tasks."""
    
    def setUp(self):
        """Set up test data."""
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        
        self.transaction = Transaction.objects.create(
            transaction_id='test_txn_001',
            user_id=self.user.id,
            amount=100.00,
            currency='USD',
            merchant='Test Merchant',
            status='completed'
        )
        
        self.alert = Alert.objects.create(
            alert_type='test_alert',
            severity='medium',
            transaction=self.transaction,
            fraud_score=75.0,
            title='Test Alert',
            message='Test message',
            owner=self.user
        )
    
    def test_cleanup_failed_deliveries(self):
        """Test cleanup of old failed deliveries."""
        # Create old failed delivery
        old_delivery = NotificationDelivery.objects.create(
            alert=self.alert,
            channel_type='email',
            recipient='test@example.com',
            status='failed',
            attempts=3,
            last_attempt_at=timezone.now() - timedelta(days=8)
        )
        
        # Create recent failed delivery (should not be cleaned up)
        recent_delivery = NotificationDelivery.objects.create(
            alert=self.alert,
            channel_type='sms',
            recipient='+1234567890',
            status='failed',
            attempts=3,
            last_attempt_at=timezone.now() - timedelta(days=1)
        )
        
        # Create old successful delivery (should be deleted)
        old_successful = NotificationDelivery.objects.create(
            alert=self.alert,
            channel_type='slack',
            recipient='@user',
            status='delivered',
            created_at=timezone.now() - timedelta(days=31)
        )
        
        result = cleanup_failed_deliveries.apply()
        
        self.assertTrue(result.successful())
        result_data = result.result
        self.assertEqual(result_data['status'], 'success')
        self.assertEqual(result_data['dead_letter_count'], 1)
        self.assertEqual(result_data['deleted_count'], 1)
        
        # Verify cleanup results
        old_delivery.refresh_from_db()
        self.assertEqual(old_delivery.status, 'dead_letter')
        
        recent_delivery.refresh_from_db()
        self.assertEqual(recent_delivery.status, 'failed')  # Unchanged
        
        # Old successful delivery should be deleted
        self.assertFalse(
            NotificationDelivery.objects.filter(id=old_successful.id).exists()
        )
    
    @patch('django.db.connection')
    @patch('django.core.cache.cache')
    def test_health_check_task(self, mock_cache, mock_connection):
        """Test system health check task."""
        # Mock successful database connection
        mock_cursor = Mock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        
        # Mock successful cache operation
        mock_cache.set.return_value = True
        mock_cache.get.return_value = 'ok'
        
        result = health_check_task.apply()
        
        self.assertTrue(result.successful())
        result_data = result.result
        self.assertTrue(result_data['database'])
        self.assertTrue(result_data['cache'])
        self.assertIn('queues', result_data)
        self.assertIn('timestamp', result_data)


@override_settings(CELERY_TASK_ALWAYS_EAGER=True)
class IntegrationTest(TestCase):
    """Integration tests for complete alert processing workflow."""
    
    def setUp(self):
        """Set up test data."""
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        
        self.transaction = Transaction.objects.create(
            transaction_id='test_txn_001',
            user_id=self.user.id,
            amount=100.00,
            currency='USD',
            merchant='Test Merchant',
            status='completed'
        )
        
        self.alert_rule = AlertRule.objects.create(
            name='Integration Test Rule',
            alert_type='integration_test',
            severity='high',
            action='alert',
            fraud_score_threshold=80.0,
            notification_channels=['email'],
            owner=self.user
        )
    
    @patch('apps.alerts.channels.factory.NotificationChannelFactory.get_channel')
    def test_complete_alert_processing_workflow(self, mock_get_channel):
        """Test complete workflow from alert creation to notification delivery."""
        # Mock successful email channel
        mock_channel = Mock()
        mock_result = Mock()
        mock_result.success = True
        mock_result.delivered = True
        mock_result.external_id = 'email_123'
        mock_result.error_message = None
        mock_result.should_retry = False
        
        mock_channel.send_notification.return_value = mock_result
        mock_get_channel.return_value = mock_channel
        
        # Create alert
        alert = Alert.objects.create(
            alert_type='integration_test',
            severity='high',
            transaction=self.transaction,
            fraud_score=85.0,
            title='Integration Test Alert',
            message='Test complete workflow',
            rule_triggered=self.alert_rule,
            owner=self.user
        )
        
        # Process the alert
        with patch('apps.alerts.channels.router.AlertRouter.route_alert') as mock_router:
            mock_router.return_value = [
                {
                    'channel_type': 'email',
                    'recipient': 'test@example.com',
                    'priority': 'high'
                }
            ]
            
            result = process_high_priority_alert.apply(args=[str(alert.id)])
            
            self.assertTrue(result.successful())
            result_data = result.result
            self.assertEqual(result_data['status'], 'processed')
            self.assertTrue(result_data['success'])
            
            # Verify alert status updated
            alert.refresh_from_db()
            self.assertEqual(alert.status, 'sent')
            
            # Verify notification delivery was created and processed
            delivery = NotificationDelivery.objects.filter(alert=alert).first()
            self.assertIsNotNone(delivery)
            self.assertEqual(delivery.status, 'delivered')
            self.assertEqual(delivery.external_id, 'email_123')
            self.assertIsNotNone(delivery.delivered_at)