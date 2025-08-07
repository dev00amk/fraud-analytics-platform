"""
Factory for creating notification channel instances.
"""

import logging
from typing import Dict, Optional

from .base import NotificationChannel

logger = logging.getLogger(__name__)


class NotificationChannelFactory:
    """Factory class for creating notification channel instances."""
    
    _channels: Dict[str, type] = {}
    
    @classmethod
    def register_channel(cls, channel_type: str, channel_class: type):
        """Register a notification channel class."""
        cls._channels[channel_type] = channel_class
        logger.info(f"Registered notification channel: {channel_type}")
    
    @classmethod
    def get_channel(cls, channel_type: str) -> Optional[NotificationChannel]:
        """Get a notification channel instance by type."""
        channel_class = cls._channels.get(channel_type)
        if not channel_class:
            logger.error(f"Unknown notification channel type: {channel_type}")
            return None
        
        try:
            return channel_class()
        except Exception as e:
            logger.error(f"Error creating channel {channel_type}: {e}")
            return None
    
    @classmethod
    def get_available_channels(cls) -> list:
        """Get list of available channel types."""
        return list(cls._channels.keys())


# Auto-register available channels
def _register_default_channels():
    """Register default notification channels."""
    try:
        from .email import EmailChannel
        NotificationChannelFactory.register_channel('email', EmailChannel)
    except ImportError:
        logger.warning("Email channel not available")
    
    try:
        from .sms import SMSChannel
        NotificationChannelFactory.register_channel('sms', SMSChannel)
    except ImportError:
        logger.warning("SMS channel not available")
    
    try:
        from .slack import SlackChannel
        NotificationChannelFactory.register_channel('slack', SlackChannel)
    except ImportError:
        logger.warning("Slack channel not available")
    
    try:
        from .webhook import WebhookChannel
        NotificationChannelFactory.register_channel('webhook', WebhookChannel)
    except ImportError:
        logger.warning("Webhook channel not available")
    
    try:
        from .teams import TeamsChannel
        NotificationChannelFactory.register_channel('teams', TeamsChannel)
    except ImportError:
        logger.warning("Teams channel not available")


# Register channels on import
_register_default_channels()