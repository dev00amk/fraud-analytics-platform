"""
Webhook notification channel implementation (placeholder).
"""

import logging
from typing import Dict, Any, Optional

from .base import NotificationChannel, Notification, DeliveryResult, DeliveryStatus

logger = logging.getLogger(__name__)


class WebhookChannel(NotificationChannel):
    """Webhook notification channel (placeholder implementation)."""
    
    @property
    def channel_type(self) -> str:
        return 'webhook'
    
    @property
    def supports_rich_content(self) -> bool:
        return True
    
    @property
    def max_message_length(self) -> Optional[int]:
        return None  # No limit for webhooks
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate webhook channel configuration."""
        # TODO: Implement webhook URL and authentication validation
        return True
    
    async def send(self, notification: Notification) -> DeliveryResult:
        """Send webhook notification (placeholder)."""
        logger.warning("Webhook channel not fully implemented - notification not sent")
        return self._create_delivery_result(
            success=False,
            status=DeliveryStatus.FAILED,
            message="Webhook channel not implemented",
            error_code="NOT_IMPLEMENTED"
        )
    
    async def get_delivery_status(self, external_id: str) -> DeliveryStatus:
        """Get webhook delivery status (placeholder)."""
        return DeliveryStatus.FAILED