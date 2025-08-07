"""
Notification template formatting and rendering system.

This module provides functionality for rendering notification templates
with dynamic content and channel-specific formatting.
"""

import logging
import re
from datetime import datetime
from typing import Dict, Any, Optional
from django.template import Template, Context
from django.template.loader import get_template
from django.utils.html import strip_tags
from django.utils.safestring import mark_safe

from ..models import NotificationTemplate, Alert

logger = logging.getLogger(__name__)


class NotificationFormatter:
    """
    Handles formatting and rendering of notification templates.
    
    This class provides methods to render notification templates
    with dynamic content and apply channel-specific formatting.
    """
    
    def __init__(self):
        self._template_cache = {}
    
    def format_notification(
        self,
        alert: Alert,
        channel_type: str,
        recipient: str,
        template_override: Optional[NotificationTemplate] = None
    ) -> Dict[str, str]:
        """
        Format a notification for the specified channel and recipient.
        
        Args:
            alert: Alert instance to format
            channel_type: Target notification channel type
            recipient: Notification recipient
            template_override: Optional template to use instead of default
            
        Returns:
            Dict[str, str]: Dictionary with 'subject' and 'body' keys
            
        Raises:
            ValueError: If no suitable template is found
        """
        # Get the appropriate template
        template = template_override or self._get_template(alert.alert_type, channel_type)
        if not template:
            raise ValueError(
                f"No template found for alert_type='{alert.alert_type}' "
                f"and channel_type='{channel_type}'"
            )
        
        # Prepare template context
        context = self._build_template_context(alert, recipient)
        
        # Render subject and body
        subject = self._render_template_string(template.subject_template, context)
        body = self._render_template_string(template.body_template, context)
        
        # Apply channel-specific formatting
        formatted_body = self._apply_channel_formatting(body, channel_type, template.formatting_options)
        
        return {
            'subject': subject,
            'body': formatted_body
        }
    
    def _get_template(self, alert_type: str, channel_type: str) -> Optional[NotificationTemplate]:
        """
        Get the appropriate template for the alert type and channel.
        
        Args:
            alert_type: Type of alert
            channel_type: Notification channel type
            
        Returns:
            NotificationTemplate: Template instance, or None if not found
        """
        cache_key = f"{alert_type}:{channel_type}"
        
        if cache_key in self._template_cache:
            return self._template_cache[cache_key]
        
        # Try to find a specific template for this alert type and channel
        template = NotificationTemplate.objects.filter(
            alert_type=alert_type,
            channel_type=channel_type,
            is_default=True
        ).first()
        
        # Fall back to a generic template for this channel
        if not template:
            template = NotificationTemplate.objects.filter(
                alert_type='generic',
                channel_type=channel_type,
                is_default=True
            ).first()
        
        # Cache the result (including None)
        self._template_cache[cache_key] = template
        
        return template
    
    def _build_template_context(self, alert: Alert, recipient: str) -> Dict[str, Any]:
        """
        Build the template context with alert data and helper variables.
        
        Args:
            alert: Alert instance
            recipient: Notification recipient
            
        Returns:
            Dict[str, Any]: Template context dictionary
        """
        # Get transaction data
        transaction = alert.transaction
        
        # Build context with all available data
        context = {
            # Alert data
            'alert': {
                'id': str(alert.id),
                'type': alert.alert_type,
                'severity': alert.severity,
                'title': alert.title,
                'message': alert.message,
                'fraud_score': alert.fraud_score,
                'risk_factors': alert.risk_factors,
                'context_data': alert.context_data,
                'created_at': alert.created_at,
                'status': alert.status,
            },
            
            # Transaction data
            'transaction': {
                'id': transaction.transaction_id,
                'amount': float(transaction.amount),
                'currency': transaction.currency,
                'merchant': transaction.merchant_id,
                'timestamp': transaction.timestamp,
                'status': transaction.status,
                'payment_method': getattr(transaction, 'payment_method', 'Unknown'),
                'location': getattr(transaction, 'location', 'Unknown'),
            },
            
            # User data (if available)
            'user': self._get_user_context(transaction),
            
            # Recipient and formatting
            'recipient': recipient,
            'current_time': datetime.now(),
            
            # Helper functions
            'format_currency': self._format_currency,
            'format_datetime': self._format_datetime,
            'severity_emoji': self._get_severity_emoji(alert.severity),
            'urgency_text': self._get_urgency_text(alert.severity),
        }
        
        return context
    
    def _get_user_context(self, transaction) -> Dict[str, Any]:
        """
        Extract user context from transaction data.
        
        Args:
            transaction: Transaction instance
            
        Returns:
            Dict[str, Any]: User context data
        """
        user_data = {
            'id': getattr(transaction, 'user_id', 'Unknown'),
            'email': getattr(transaction, 'user_email', 'Unknown'),
            'name': getattr(transaction, 'user_name', 'Unknown'),
        }
        
        # Add additional user data from transaction metadata if available
        if hasattr(transaction, 'metadata') and transaction.metadata:
            user_data.update(transaction.metadata.get('user', {}))
        
        return user_data
    
    def _render_template_string(self, template_string: str, context: Dict[str, Any]) -> str:
        """
        Render a template string with the given context.
        
        Args:
            template_string: Django template string
            context: Template context
            
        Returns:
            str: Rendered template
        """
        if not template_string:
            return ""
        
        try:
            template = Template(template_string)
            django_context = Context(context)
            return template.render(django_context)
        except Exception as e:
            logger.error(f"Template rendering error: {e}")
            logger.error(f"Template: {template_string}")
            # Return a safe fallback
            return f"Alert: {context.get('alert', {}).get('title', 'Fraud Alert')}"
    
    def _apply_channel_formatting(
        self,
        content: str,
        channel_type: str,
        formatting_options: Dict[str, Any]
    ) -> str:
        """
        Apply channel-specific formatting to content.
        
        Args:
            content: Content to format
            channel_type: Target channel type
            formatting_options: Channel-specific formatting options
            
        Returns:
            str: Formatted content
        """
        if channel_type == 'email':
            return self._format_for_email(content, formatting_options)
        elif channel_type == 'sms':
            return self._format_for_sms(content, formatting_options)
        elif channel_type == 'slack':
            return self._format_for_slack(content, formatting_options)
        elif channel_type == 'teams':
            return self._format_for_teams(content, formatting_options)
        else:
            return content
    
    def _format_for_email(self, content: str, options: Dict[str, Any]) -> str:
        """Format content for email delivery."""
        # Convert line breaks to HTML if HTML format is enabled
        if options.get('html_format', False):
            content = content.replace('\n', '<br>\n')
            return mark_safe(content)
        return content
    
    def _format_for_sms(self, content: str, options: Dict[str, Any]) -> str:
        """Format content for SMS delivery."""
        # Strip HTML tags and limit length
        content = strip_tags(content)
        max_length = options.get('max_length', 160)
        
        if len(content) > max_length:
            content = content[:max_length - 3] + "..."
        
        return content
    
    def _format_for_slack(self, content: str, options: Dict[str, Any]) -> str:
        """Format content for Slack delivery."""
        # Convert basic formatting to Slack markdown
        # Use a placeholder to avoid conflicts between bold and italic processing
        placeholder = "___SLACK_BOLD_PLACEHOLDER___"
        
        # First handle bold (double asterisk) with placeholder
        content = re.sub(r'\*\*(.*?)\*\*', rf'{placeholder}\1{placeholder}', content)
        
        # Then handle italic (single asterisk)
        content = re.sub(r'\*([^*]+?)\*', r'_\1_', content)
        
        # Finally replace placeholders with Slack bold format
        content = content.replace(placeholder, '*')
        
        return content
    
    def _format_for_teams(self, content: str, options: Dict[str, Any]) -> str:
        """Format content for Microsoft Teams delivery."""
        # Teams supports basic markdown
        return content
    
    def _format_currency(self, amount: float, currency: str = 'USD') -> str:
        """Format currency amount."""
        if currency == 'USD':
            return f"${amount:,.2f}"
        else:
            return f"{amount:,.2f} {currency}"
    
    def _format_datetime(self, dt: datetime, format_str: str = '%Y-%m-%d %H:%M:%S UTC') -> str:
        """Format datetime."""
        return dt.strftime(format_str)
    
    def _get_severity_emoji(self, severity: str) -> str:
        """Get emoji for alert severity."""
        emoji_map = {
            'low': '🟡',
            'medium': '🟠', 
            'high': '🔴',
            'critical': '🚨'
        }
        return emoji_map.get(severity, '⚠️')
    
    def _get_urgency_text(self, severity: str) -> str:
        """Get urgency text for alert severity."""
        urgency_map = {
            'low': 'Low Priority',
            'medium': 'Medium Priority',
            'high': 'High Priority - Immediate Attention Required',
            'critical': 'CRITICAL - Immediate Action Required'
        }
        return urgency_map.get(severity, 'Alert')
    
    def clear_template_cache(self):
        """Clear the template cache."""
        self._template_cache.clear()
        logger.info("Template cache cleared")


# Global formatter instance
notification_formatter = NotificationFormatter()