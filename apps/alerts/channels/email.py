"""
Email notification channel implementation.
"""

import logging
from typing import Dict, Any, Optional
from django.core.mail import send_mail
from django.conf import settings

from .base import NotificationChannel, Notification, DeliveryResult, DeliveryStatus, NotificationError

logger = logging.getLogger(__name__)


class EmailChannel(NotificationChannel):
    """Email notification channel using Django's email backend."""
    
    @property
    def channel_type(self) -> str:
        return 'email'
    
    @property
    def supports_rich_content(self) -> bool:
        return True
    
    @property
    def max_message_length(self) -> Optional[int]:
        return None  # No limit for email
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate email channel configuration."""
        # Email channel uses Django's email settings, so minimal config needed
        return True
    
    async def send(self, notification: Notification) -> DeliveryResult:
        """Send email notification."""
        try:
            # Validate recipient email
            if not self._is_valid_email(notification.recipient):
                return self._create_delivery_result(
                    success=False,
                    status=DeliveryStatus.FAILED,
                    message=f"Invalid email address: {notification.recipient}",
                    error_code="INVALID_EMAIL"
                )
            
            # Format message for email
            formatted_notification = self.format_message(notification)
            
            # Send email using Django's send_mail
            try:
                send_mail(
                    subject=formatted_notification.subject,
                    message=formatted_notification.body,
                    from_email=settings.DEFAULT_FROM_EMAIL if hasattr(settings, 'DEFAULT_FROM_EMAIL') else 'noreply@fraudplatform.com',
                    recipient_list=[formatted_notification.recipient],
                    fail_silently=False
                )
                
                return self._create_delivery_result(
                    success=True,
                    status=DeliveryStatus.SENT,
                    message="Email sent successfully",
                    external_id=f"email_{notification.alert_id}_{hash(notification.recipient)}"
                )
                
            except Exception as e:
                logger.error(f"Failed to send email to {notification.recipient}: {e}")
                return self._create_delivery_result(
                    success=False,
                    status=DeliveryStatus.FAILED,
                    message=f"Email delivery failed: {str(e)}",
                    error_code="DELIVERY_FAILED",
                    retry_after=300  # Retry after 5 minutes
                )
                
        except Exception as e:
            logger.error(f"Unexpected error in email channel: {e}")
            return self._create_delivery_result(
                success=False,
                status=DeliveryStatus.FAILED,
                message=f"Unexpected error: {str(e)}",
                error_code="UNEXPECTED_ERROR"
            )
    
    async def get_delivery_status(self, external_id: str) -> DeliveryStatus:
        """Get delivery status for email (limited tracking available)."""
        # Email delivery status is limited without external service integration
        # For now, assume sent emails are delivered
        return DeliveryStatus.DELIVERED
    
    def _is_valid_email(self, email: str) -> bool:
        """Basic email validation."""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def format_message(self, notification: Notification) -> Notification:
        """Format email message with HTML support if needed."""
        formatted = super().format_message(notification)
        
        # Add email-specific formatting
        if not formatted.subject.startswith('[FRAUD ALERT]'):
            formatted.subject = f"[FRAUD ALERT] {formatted.subject}"
        
        # Add footer to body
        formatted.body += "\n\n---\nThis is an automated fraud alert from the Fraud Analytics Platform."
        
        return formatted