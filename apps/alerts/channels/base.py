"""
Base classes and interfaces for notification channels.

This module defines the abstract base class and common interfaces
that all notification channels must implement.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
from uuid import UUID

logger = logging.getLogger(__name__)


class DeliveryStatus(Enum):
    """Enumeration of possible delivery statuses."""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class DeliveryResult:
    """
    Result of a notification delivery attempt.
    
    Attributes:
        success: Whether the delivery was successful
        status: Current delivery status
        external_id: External service's delivery ID (if available)
        message: Human-readable status message
        error_code: Error code for failed deliveries
        retry_after: Seconds to wait before retry (for temporary failures)
        metadata: Additional channel-specific data
    """
    success: bool
    status: DeliveryStatus
    external_id: Optional[str] = None
    message: str = ""
    error_code: Optional[str] = None
    retry_after: Optional[int] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class NotificationError(Exception):
    """Base exception for notification delivery errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, metadata: Dict[str, Any] = None):
        super().__init__(message)
        self.error_code = error_code
        self.metadata = metadata or {}


class TemporaryDeliveryError(NotificationError):
    """Exception for temporary delivery failures that should be retried."""
    
    def __init__(self, message: str, retry_after: int = 60, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class PermanentDeliveryError(NotificationError):
    """Exception for permanent delivery failures that should not be retried."""
    pass


@dataclass
class Notification:
    """
    Represents a notification to be sent through a channel.
    
    Attributes:
        alert_id: UUID of the associated alert
        recipient: Target recipient (email, phone, webhook URL, etc.)
        subject: Notification subject/title
        body: Main notification content
        priority: Notification priority (1=highest, 5=lowest)
        metadata: Additional channel-specific data
        template_data: Data for template rendering
    """
    alert_id: UUID
    recipient: str
    subject: str
    body: str
    priority: int = 3
    metadata: Dict[str, Any] = None
    template_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.template_data is None:
            self.template_data = {}


class NotificationChannel(ABC):
    """
    Abstract base class for all notification channels.
    
    This class defines the interface that all notification channels
    must implement to integrate with the alerting system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the notification channel with configuration.
        
        Args:
            config: Channel-specific configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self.validate_config(config)
        self._setup_channel()
    
    @property
    @abstractmethod
    def channel_type(self) -> str:
        """Return the channel type identifier (e.g., 'email', 'sms')."""
        pass
    
    @property
    @abstractmethod
    def supports_rich_content(self) -> bool:
        """Return True if the channel supports rich content (HTML, attachments, etc.)."""
        pass
    
    @property
    @abstractmethod
    def max_message_length(self) -> Optional[int]:
        """Return maximum message length, or None if unlimited."""
        pass
    
    @abstractmethod
    async def send(self, notification: Notification) -> DeliveryResult:
        """
        Send a notification through this channel.
        
        Args:
            notification: The notification to send
            
        Returns:
            DeliveryResult: Result of the delivery attempt
            
        Raises:
            NotificationError: If delivery fails
        """
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate the channel configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            bool: True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    async def get_delivery_status(self, external_id: str) -> DeliveryStatus:
        """
        Get the current delivery status for a previously sent notification.
        
        Args:
            external_id: External service's delivery ID
            
        Returns:
            DeliveryStatus: Current delivery status
            
        Raises:
            NotificationError: If status cannot be retrieved
        """
        pass
    
    def send_notification(self, delivery):
        """
        Send notification using NotificationDelivery model (sync wrapper).
        
        This method provides compatibility with the task system that uses
        NotificationDelivery models instead of Notification objects.
        
        Args:
            delivery: NotificationDelivery model instance
            
        Returns:
            DeliveryResult: Result of the delivery attempt
        """
        try:
            # Convert NotificationDelivery to Notification
            from ..models import NotificationTemplate
            
            # Get template for this channel and alert type
            template = NotificationTemplate.objects.filter(
                channel_type=delivery.channel_type,
                alert_type=delivery.alert.alert_type,
                is_default=True
            ).first()
            
            # Create notification object
            notification = Notification(
                alert_id=delivery.alert.id,
                recipient=delivery.recipient,
                subject=self._render_subject(delivery.alert, template),
                body=self._render_body(delivery.alert, template),
                priority=self._get_priority_from_severity(delivery.alert.severity),
                metadata={'delivery_id': str(delivery.id)},
                template_data=self._get_template_data(delivery.alert)
            )
            
            # Use asyncio to run the async send method
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            return loop.run_until_complete(self.send(notification))
            
        except Exception as e:
            logger.error(f"Error in send_notification: {e}")
            return DeliveryResult(
                success=False,
                status=DeliveryStatus.FAILED,
                message=str(e),
                error_code="SEND_ERROR"
            )
    
    def validate_recipient(self, recipient: str) -> bool:
        """
        Validate a recipient address for this channel.
        
        Args:
            recipient: Recipient address to validate
            
        Returns:
            bool: True if recipient is valid
        """
        return bool(recipient and recipient.strip())
    
    def format_message(self, notification: Notification) -> Notification:
        """
        Format the notification message for this channel.
        
        This method can be overridden by subclasses to apply
        channel-specific formatting (e.g., truncation, markup).
        
        Args:
            notification: Original notification
            
        Returns:
            Notification: Formatted notification
        """
        formatted = Notification(
            alert_id=notification.alert_id,
            recipient=notification.recipient,
            subject=notification.subject,
            body=notification.body,
            priority=notification.priority,
            metadata=notification.metadata.copy(),
            template_data=notification.template_data.copy()
        )
        
        # Apply message length limits if applicable
        if self.max_message_length:
            if len(formatted.body) > self.max_message_length:
                truncated_length = self.max_message_length - 3  # Account for "..."
                formatted.body = formatted.body[:truncated_length] + "..."
                logger.warning(
                    f"Message truncated for {self.channel_type} channel: "
                    f"original={len(notification.body)}, truncated={len(formatted.body)}"
                )
        
        return formatted
    
    def _setup_channel(self):
        """
        Perform any additional setup after configuration validation.
        
        This method can be overridden by subclasses to initialize
        connections, authenticate with services, etc.
        """
        pass
    
    def _create_delivery_result(
        self,
        success: bool,
        status: DeliveryStatus,
        message: str = "",
        external_id: Optional[str] = None,
        error_code: Optional[str] = None,
        retry_after: Optional[int] = None,
        **metadata
    ) -> DeliveryResult:
        """
        Helper method to create a DeliveryResult with consistent logging.
        
        Args:
            success: Whether delivery was successful
            status: Delivery status
            message: Status message
            external_id: External service delivery ID
            error_code: Error code for failures
            retry_after: Retry delay for temporary failures
            **metadata: Additional metadata
            
        Returns:
            DeliveryResult: Formatted delivery result
        """
        result = DeliveryResult(
            success=success,
            status=status,
            external_id=external_id,
            message=message,
            error_code=error_code,
            retry_after=retry_after,
            metadata=metadata
        )
        
        log_level = logging.INFO if success else logging.WARNING
        logger.log(
            log_level,
            f"{self.channel_type} delivery result: success={success}, "
            f"status={status.value}, external_id={external_id}, message={message}"
        )
        
        return result
    
    def _render_subject(self, alert, template):
        """Render notification subject from template."""
        if template and template.subject_template:
            return self._render_template(template.subject_template, alert)
        else:
            return f"Fraud Alert: {alert.alert_type} - {alert.severity.upper()}"
    
    def _render_body(self, alert, template):
        """Render notification body from template."""
        if template and template.body_template:
            return self._render_template(template.body_template, alert)
        else:
            return self._get_default_body(alert)
    
    def _render_template(self, template_text, alert):
        """Render template with alert data."""
        try:
            template_data = self._get_template_data(alert)
            return template_text.format(**template_data)
        except Exception as e:
            logger.error(f"Error rendering template: {e}")
            return self._get_default_body(alert)
    
    def _get_template_data(self, alert):
        """Get template data for rendering."""
        return {
            'alert_id': str(alert.id),
            'alert_type': alert.alert_type,
            'severity': alert.severity,
            'title': alert.title,
            'message': alert.message,
            'fraud_score': alert.fraud_score,
            'transaction_id': alert.transaction.transaction_id if alert.transaction else 'N/A',
            'transaction_amount': alert.transaction.amount if alert.transaction else 'N/A',
            'created_at': alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC'),
            'risk_factors': ', '.join(alert.risk_factors.keys()) if alert.risk_factors else 'None'
        }
    
    def _get_default_body(self, alert):
        """Get default notification body."""
        return f"""
Fraud Alert: {alert.title}

Severity: {alert.severity.upper()}
Alert Type: {alert.alert_type}
Fraud Score: {alert.fraud_score}
Transaction ID: {alert.transaction.transaction_id if alert.transaction else 'N/A'}
Created: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}

Message: {alert.message}

Risk Factors: {', '.join(alert.risk_factors.keys()) if alert.risk_factors else 'None'}

Please investigate this alert immediately.
        """.strip()
    
    def _get_priority_from_severity(self, severity):
        """Convert alert severity to notification priority."""
        priority_mapping = {
            'critical': 1,
            'high': 2,
            'medium': 3,
            'low': 4
        }
        return priority_mapping.get(severity, 3)


class ChannelRegistry:
    """
    Registry for managing available notification channels.
    
    This class maintains a registry of available channel types
    and provides factory methods for creating channel instances.
    """
    
    def __init__(self):
        self._channels: Dict[str, type] = {}
    
    def register(self, channel_type: str, channel_class: type):
        """
        Register a notification channel class.
        
        Args:
            channel_type: Channel type identifier
            channel_class: Channel class (must inherit from NotificationChannel)
            
        Raises:
            ValueError: If channel_class is not a NotificationChannel subclass
        """
        if not issubclass(channel_class, NotificationChannel):
            raise ValueError(f"Channel class must inherit from NotificationChannel")
        
        self._channels[channel_type] = channel_class
        logger.info(f"Registered notification channel: {channel_type}")
    
    def get_channel_class(self, channel_type: str) -> Optional[type]:
        """
        Get a registered channel class by type.
        
        Args:
            channel_type: Channel type identifier
            
        Returns:
            type: Channel class, or None if not found
        """
        return self._channels.get(channel_type)
    
    def create_channel(self, channel_type: str, config: Dict[str, Any]) -> NotificationChannel:
        """
        Create a channel instance.
        
        Args:
            channel_type: Channel type identifier
            config: Channel configuration
            
        Returns:
            NotificationChannel: Channel instance
            
        Raises:
            ValueError: If channel type is not registered
        """
        channel_class = self.get_channel_class(channel_type)
        if not channel_class:
            raise ValueError(f"Unknown channel type: {channel_type}")
        
        return channel_class(config)
    
    def list_channels(self) -> List[str]:
        """
        Get a list of registered channel types.
        
        Returns:
            List[str]: List of channel type identifiers
        """
        return list(self._channels.keys())


# Global channel registry instance
channel_registry = ChannelRegistry()