"""
SMS notification channel implementation (placeholder).
"""

import logging
from typing import Dict, Any, Optional

from .base import NotificationChannel, Notification, DeliveryResult, DeliveryStatus

logger = logging.getLogger(__name__)


class SMSChannel(NotificationChannel):
    """SMS notification channel (placeholder implementation)."""
    
    @property
    def channel_type(self) -> str:
        return 'sms'
    
    @property
    def supports_rich_content(self) -> bool:
        return False
    
    @property
    def max_message_length(self) -> Optional[int]:
        return 160  # Standard SMS length
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate SMS channel configuration."""
        # TODO: Implement Twilio configuration validation
        return True
    
    async def send(self, notification: Notification) -> DeliveryResult:
        """Send SMS notification (placeholder)."""
        logger.warning("SMS channel not fully implemented - notification not sent")
        return self._create_delivery_result(
            success=False,
            status=DeliveryStatus.FAILED,
            message="SMS channel not implemented",
            error_code="NOT_IMPLEMENTED"
        )
    
    async def get_delivery_status(self, external_id: str) -> DeliveryStatus:
        """Get SMS delivery status (placeholder)."""
        return DeliveryStatus.FAILED