"""
Unit tests for notification channel base classes and interfaces.

This module tests the core notification channel functionality including
base classes, delivery results, and template formatting.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from uuid import uuid4

from django.test import TestCase
from django.contrib.auth import get_user_model

from apps.alerts.channels.base import (
    NotificationChannel,
    DeliveryResult,
    DeliveryStatus,
    Notification,
    NotificationError,
    TemporaryDeliveryError,
    PermanentDeliveryError,
    ChannelRegistry,
    channel_registry
)
from apps.alerts.channels.template import NotificationFormatter, notification_formatter
from apps.alerts.models import Alert, NotificationTemplate
from apps.transactions.models import Transaction

User = get_user_model()


class MockNotificationChannel(NotificationChannel):
    """Mock notification channel for testing."""
    
    @property
    def channel_type(self) -> str:
        return "mock"
    
    @property
    def supports_rich_content(self) -> bool:
        return True
    
    @property
    def max_message_length(self) -> int:
        return 1000
    
    async def send(self, notification: Notification) -> DeliveryResult:
        if notification.recipient == "fail@example.com":
            raise TemporaryDeliveryError("Mock failure", retry_after=60)
        
        return DeliveryResult(
            success=True,
            status=DeliveryStatus.SENT,
            external_id="mock-123",
            message="Mock delivery successful"
        )
    
    def validate_config(self, config: dict) -> bool:
        required_keys = ['api_key']
        if not all(key in config for key in required_keys):
            raise ValueError("Missing required configuration keys")
        return True
    
    async def get_delivery_status(self, external_id: str) -> DeliveryStatus:
        if external_id == "mock-123":
            return DeliveryStatus.DELIVERED
        return DeliveryStatus.FAILED


class TestDeliveryResult(TestCase):
    """Test DeliveryResult dataclass."""
    
    def test_delivery_result_creation(self):
        """Test creating a DeliveryResult instance."""
        result = DeliveryResult(
            success=True,
            status=DeliveryStatus.SENT,
            external_id="test-123",
            message="Test message"
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.status, DeliveryStatus.SENT)
        self.assertEqual(result.external_id, "test-123")
        self.assertEqual(result.message, "Test message")
        self.assertIsNone(result.error_code)
        self.assertIsNone(result.retry_after)
        self.assertEqual(result.metadata, {})
    
    def test_delivery_result_with_metadata(self):
        """Test DeliveryResult with metadata."""
        metadata = {"provider": "test", "cost": 0.01}
        result = DeliveryResult(
            success=False,
            status=DeliveryStatus.FAILED,
            error_code="INVALID_RECIPIENT",
            metadata=metadata
        )
        
        self.assertFalse(result.success)
        self.assertEqual(result.status, DeliveryStatus.FAILED)
        self.assertEqual(result.error_code, "INVALID_RECIPIENT")
        self.assertEqual(result.metadata, metadata)


class TestNotificationErrors(TestCase):
    """Test notification error classes."""
    
    def test_notification_error(self):
        """Test base NotificationError."""
        error = NotificationError(
            "Test error",
            error_code="TEST_ERROR",
            metadata={"key": "value"}
        )
        
        self.assertEqual(str(error), "Test error")
        self.assertEqual(error.error_code, "TEST_ERROR")
        self.assertEqual(error.metadata, {"key": "value"})
    
    def test_temporary_delivery_error(self):
        """Test TemporaryDeliveryError."""
        error = TemporaryDeliveryError(
            "Temporary failure",
            retry_after=120,
            error_code="RATE_LIMITED"
        )
        
        self.assertEqual(str(error), "Temporary failure")
        self.assertEqual(error.retry_after, 120)
        self.assertEqual(error.error_code, "RATE_LIMITED")
    
    def test_permanent_delivery_error(self):
        """Test PermanentDeliveryError."""
        error = PermanentDeliveryError(
            "Permanent failure",
            error_code="INVALID_CONFIG"
        )
        
        self.assertEqual(str(error), "Permanent failure")
        self.assertEqual(error.error_code, "INVALID_CONFIG")


class TestNotification(TestCase):
    """Test Notification dataclass."""
    
    def test_notification_creation(self):
        """Test creating a Notification instance."""
        alert_id = uuid4()
        notification = Notification(
            alert_id=alert_id,
            recipient="test@example.com",
            subject="Test Alert",
            body="This is a test alert message"
        )
        
        self.assertEqual(notification.alert_id, alert_id)
        self.assertEqual(notification.recipient, "test@example.com")
        self.assertEqual(notification.subject, "Test Alert")
        self.assertEqual(notification.body, "This is a test alert message")
        self.assertEqual(notification.priority, 3)  # Default priority
        self.assertEqual(notification.metadata, {})
        self.assertEqual(notification.template_data, {})
    
    def test_notification_with_metadata(self):
        """Test Notification with metadata and template data."""
        alert_id = uuid4()
        metadata = {"channel": "email", "template": "fraud_alert"}
        template_data = {"user_name": "John Doe", "amount": 100.00}
        
        notification = Notification(
            alert_id=alert_id,
            recipient="test@example.com",
            subject="Test Alert",
            body="Alert for {{user_name}}",
            priority=1,
            metadata=metadata,
            template_data=template_data
        )
        
        self.assertEqual(notification.priority, 1)
        self.assertEqual(notification.metadata, metadata)
        self.assertEqual(notification.template_data, template_data)


class TestNotificationChannel(TestCase):
    """Test NotificationChannel base class."""
    
    def setUp(self):
        """Set up test data."""
        self.valid_config = {"api_key": "test-key"}
        self.invalid_config = {"invalid": "config"}
    
    def test_channel_initialization(self):
        """Test channel initialization with valid config."""
        channel = MockNotificationChannel(self.valid_config)
        
        self.assertEqual(channel.config, self.valid_config)
        self.assertEqual(channel.channel_type, "mock")
        self.assertTrue(channel.supports_rich_content)
        self.assertEqual(channel.max_message_length, 1000)
    
    def test_channel_initialization_invalid_config(self):
        """Test channel initialization with invalid config."""
        with self.assertRaises(ValueError):
            MockNotificationChannel(self.invalid_config)
    
    def test_validate_recipient(self):
        """Test recipient validation."""
        channel = MockNotificationChannel(self.valid_config)
        
        self.assertTrue(channel.validate_recipient("test@example.com"))
        self.assertTrue(channel.validate_recipient("valid-recipient"))
        self.assertFalse(channel.validate_recipient(""))
        self.assertFalse(channel.validate_recipient("   "))
        self.assertFalse(channel.validate_recipient(None))
    
    def test_format_message(self):
        """Test message formatting."""
        channel = MockNotificationChannel(self.valid_config)
        
        notification = Notification(
            alert_id=uuid4(),
            recipient="test@example.com",
            subject="Test",
            body="Short message"
        )
        
        formatted = channel.format_message(notification)
        self.assertEqual(formatted.body, "Short message")
    
    def test_format_message_truncation(self):
        """Test message truncation for length limits."""
        # Create a channel with short message limit
        class ShortMessageChannel(MockNotificationChannel):
            @property
            def max_message_length(self) -> int:
                return 20
        
        channel = ShortMessageChannel(self.valid_config)
        
        notification = Notification(
            alert_id=uuid4(),
            recipient="test@example.com",
            subject="Test",
            body="This is a very long message that should be truncated"
        )
        
        formatted = channel.format_message(notification)
        self.assertEqual(len(formatted.body), 20)
        self.assertTrue(formatted.body.endswith("..."))
    
    @pytest.mark.asyncio
    async def test_send_notification(self):
        """Test sending a notification."""
        channel = MockNotificationChannel(self.valid_config)
        
        notification = Notification(
            alert_id=uuid4(),
            recipient="test@example.com",
            subject="Test Alert",
            body="Test message"
        )
        
        result = await channel.send(notification)
        
        self.assertTrue(result.success)
        self.assertEqual(result.status, DeliveryStatus.SENT)
        self.assertEqual(result.external_id, "mock-123")
    
    @pytest.mark.asyncio
    async def test_send_notification_failure(self):
        """Test sending a notification that fails."""
        channel = MockNotificationChannel(self.valid_config)
        
        notification = Notification(
            alert_id=uuid4(),
            recipient="fail@example.com",
            subject="Test Alert",
            body="Test message"
        )
        
        with self.assertRaises(TemporaryDeliveryError) as context:
            await channel.send(notification)
        
        self.assertEqual(context.exception.retry_after, 60)
    
    @pytest.mark.asyncio
    async def test_get_delivery_status(self):
        """Test getting delivery status."""
        channel = MockNotificationChannel(self.valid_config)
        
        status = await channel.get_delivery_status("mock-123")
        self.assertEqual(status, DeliveryStatus.DELIVERED)
        
        status = await channel.get_delivery_status("unknown-id")
        self.assertEqual(status, DeliveryStatus.FAILED)


class TestChannelRegistry(TestCase):
    """Test ChannelRegistry functionality."""
    
    def setUp(self):
        """Set up test registry."""
        self.registry = ChannelRegistry()
    
    def test_register_channel(self):
        """Test registering a channel."""
        self.registry.register("mock", MockNotificationChannel)
        
        channel_class = self.registry.get_channel_class("mock")
        self.assertEqual(channel_class, MockNotificationChannel)
    
    def test_register_invalid_channel(self):
        """Test registering an invalid channel class."""
        class InvalidChannel:
            pass
        
        with self.assertRaises(ValueError):
            self.registry.register("invalid", InvalidChannel)
    
    def test_create_channel(self):
        """Test creating a channel instance."""
        self.registry.register("mock", MockNotificationChannel)
        
        config = {"api_key": "test-key"}
        channel = self.registry.create_channel("mock", config)
        
        self.assertIsInstance(channel, MockNotificationChannel)
        self.assertEqual(channel.config, config)
    
    def test_create_unknown_channel(self):
        """Test creating an unknown channel type."""
        with self.assertRaises(ValueError):
            self.registry.create_channel("unknown", {})
    
    def test_list_channels(self):
        """Test listing registered channels."""
        self.registry.register("mock1", MockNotificationChannel)
        self.registry.register("mock2", MockNotificationChannel)
        
        channels = self.registry.list_channels()
        self.assertIn("mock1", channels)
        self.assertIn("mock2", channels)


class TestNotificationFormatter(TestCase):
    """Test NotificationFormatter functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.user = User.objects.create_user(
            username="testuser",
            email="test@example.com"
        )
        
        self.transaction = Transaction.objects.create(
            transaction_id="TXN-123",
            amount=100.00,
            currency="USD",
            merchant_id="merchant-123",
            payment_method="credit_card",
            user_id="user-123",
            ip_address="192.168.1.1",
            timestamp=datetime.now(),
            owner=self.user
        )
        
        self.alert = Alert.objects.create(
            alert_type="fraud_detection",
            severity="high",
            transaction=self.transaction,
            fraud_score=0.85,
            title="High Risk Transaction",
            message="Suspicious transaction detected",
            owner=self.user,
            risk_factors={"location": "unusual", "amount": "high"}
        )
        
        self.template = NotificationTemplate.objects.create(
            name="Test Template",
            channel_type="email",
            alert_type="fraud_detection",
            subject_template="Alert: {{alert.title}}",
            body_template="Transaction {{transaction.id}} for {{transaction.amount}} flagged with score {{alert.fraud_score}}",
            is_default=True,
            owner=self.user
        )
        
        self.formatter = NotificationFormatter()
    
    def test_format_notification(self):
        """Test formatting a notification."""
        result = self.formatter.format_notification(
            self.alert,
            "email",
            "test@example.com"
        )
        
        self.assertIn("Alert: High Risk Transaction", result['subject'])
        self.assertIn("TXN-123", result['body'])
        self.assertIn("100.0", result['body'])
        self.assertIn("0.85", result['body'])
    
    def test_format_notification_no_template(self):
        """Test formatting when no template is found."""
        with self.assertRaises(ValueError):
            self.formatter.format_notification(
                self.alert,
                "unknown_channel",
                "test@example.com"
            )
    
    def test_build_template_context(self):
        """Test building template context."""
        context = self.formatter._build_template_context(
            self.alert,
            "test@example.com"
        )
        
        self.assertEqual(context['alert']['id'], str(self.alert.id))
        self.assertEqual(context['alert']['type'], "fraud_detection")
        self.assertEqual(context['alert']['severity'], "high")
        self.assertEqual(context['transaction']['id'], "TXN-123")
        self.assertEqual(context['transaction']['amount'], 100.0)
        self.assertEqual(context['recipient'], "test@example.com")
        self.assertIn('current_time', context)
    
    def test_channel_specific_formatting(self):
        """Test channel-specific formatting."""
        content = "**Bold text** and *italic text*"
        
        # Test email formatting
        email_formatted = self.formatter._apply_channel_formatting(
            content, "email", {"html_format": True}
        )
        self.assertIn("<br>", email_formatted) if "\n" in content else None
        
        # Test SMS formatting
        sms_formatted = self.formatter._apply_channel_formatting(
            content, "sms", {"max_length": 20}
        )
        self.assertLessEqual(len(sms_formatted), 20)
        
        # Test Slack formatting
        slack_formatted = self.formatter._apply_channel_formatting(
            content, "slack", {}
        )
        self.assertIn("*Bold text*", slack_formatted)
        self.assertIn("_italic text_", slack_formatted)
    
    def test_helper_functions(self):
        """Test template helper functions."""
        # Test currency formatting
        formatted = self.formatter._format_currency(1234.56, "USD")
        self.assertEqual(formatted, "$1,234.56")
        
        # Test datetime formatting
        dt = datetime(2023, 12, 25, 15, 30, 45)
        formatted = self.formatter._format_datetime(dt)
        self.assertIn("2023-12-25", formatted)
        
        # Test severity emoji
        emoji = self.formatter._get_severity_emoji("critical")
        self.assertEqual(emoji, "🚨")
        
        # Test urgency text
        urgency = self.formatter._get_urgency_text("high")
        self.assertIn("Immediate Attention", urgency)
    
    def test_template_cache(self):
        """Test template caching."""
        # First call should cache the template
        result1 = self.formatter.format_notification(
            self.alert,
            "email",
            "test@example.com"
        )
        
        # Second call should use cached template
        result2 = self.formatter.format_notification(
            self.alert,
            "email",
            "test@example.com"
        )
        
        self.assertEqual(result1, result2)
        
        # Clear cache and verify it's empty
        self.formatter.clear_template_cache()
        self.assertEqual(len(self.formatter._template_cache), 0)


class TestGlobalInstances(TestCase):
    """Test global instances and registry."""
    
    def test_global_channel_registry(self):
        """Test global channel registry instance."""
        self.assertIsInstance(channel_registry, ChannelRegistry)
    
    def test_global_notification_formatter(self):
        """Test global notification formatter instance."""
        self.assertIsInstance(notification_formatter, NotificationFormatter)