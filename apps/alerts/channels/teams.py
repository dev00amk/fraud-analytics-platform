"""
Microsoft Teams notification channel implementation (placeholder).
"""

import logging
from typing import Dict, Any, Optional

from .base import NotificationChannel, Notification, DeliveryResult, DeliveryStatus

logger = logging.getLogger(__name__)


class TeamsChannel(NotificationChannel):
    """Microsoft Teams notification channel (placeholder implementation)."""
    
    @property
    def channel_type(self) -> str:
        return 'teams'
    
    @property
    def supports_rich_content(self) -> bool:
        return True
    
    @property
    def max_message_length(self) -> Optional[int]:
        return 4000  # Teams message limit
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate Teams channel configuration."""
        # TODO: Implement Teams API configuration validation
        return True
    
    async def send(self, notification: Notification) -> DeliveryResult:
        """Send Teams notification (placeholder)."""
        logger.warning("Teams channel not fully implemented - notification not sent")
        return self._create_delivery_result(
            success=False,
            status=DeliveryStatus.FAILED,
            message="Teams channel not implemented",
            error_code="NOT_IMPLEMENTED"
        )
    
    async def get_delivery_status(self, external_id: str) -> DeliveryStatus:
        """Get Teams delivery status (placeholder)."""
        return DeliveryStatus.FAILED