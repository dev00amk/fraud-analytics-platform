"""
Notification channels package for the fraud alerting system.

This package provides the base classes and interfaces for implementing
various notification channels (email, SMS, Slack, webhook, etc.).
"""

from .base import (
    NotificationChannel,
    DeliveryResult,
    DeliveryStatus,
    Notification,
    NotificationError,
    TemporaryDeliveryError,
    PermanentDeliveryError,
    ChannelRegistry,
    channel_registry,
)
from .template import NotificationFormatter, notification_formatter

__all__ = [
    'NotificationChannel',
    'DeliveryResult', 
    'DeliveryStatus',
    'Notification',
    'NotificationError',
    'TemporaryDeliveryError',
    'PermanentDeliveryError',
    'ChannelRegistry',
    'channel_registry',
    'NotificationFormatter',
    'notification_formatter',
]