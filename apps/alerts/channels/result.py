"""
Notification delivery result classes.
"""

from typing import Optional


class DeliveryResult:
    """Result of a notification delivery attempt."""
    
    def __init__(
        self,
        success: bool = False,
        delivered: bool = False,
        external_id: Optional[str] = None,
        error_message: Optional[str] = None,
        should_retry: bool = True
    ):
        self.success = success
        self.delivered = delivered
        self.external_id = external_id
        self.error_message = error_message
        self.should_retry = should_retry
    
    @classmethod
    def success_result(cls, external_id: Optional[str] = None, delivered: bool = True):
        """Create a successful delivery result."""
        return cls(
            success=True,
            delivered=delivered,
            external_id=external_id,
            should_retry=False
        )
    
    @classmethod
    def failure_result(cls, error_message: str, should_retry: bool = True):
        """Create a failed delivery result."""
        return cls(
            success=False,
            delivered=False,
            error_message=error_message,
            should_retry=should_retry
        )
    
    def __str__(self):
        if self.success:
            return f"DeliveryResult(success=True, delivered={self.delivered}, external_id={self.external_id})"
        else:
            return f"DeliveryResult(success=False, error='{self.error_message}', should_retry={self.should_retry})"


class DeliveryStatus:
    """Status of a notification delivery."""
    
    def __init__(
        self,
        status: str,
        delivered_at: Optional[str] = None,
        error_message: Optional[str] = None,
        external_data: Optional[dict] = None
    ):
        self.status = status
        self.delivered_at = delivered_at
        self.error_message = error_message
        self.external_data = external_data or {}
    
    def __str__(self):
        return f"DeliveryStatus(status='{self.status}', delivered_at={self.delivered_at})"