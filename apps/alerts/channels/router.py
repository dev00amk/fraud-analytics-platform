"""
Alert routing engine that determines notification channels and recipients.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.utils import timezone

from ..models import Alert, AlertRule, NotificationTemplate

User = get_user_model()
logger = logging.getLogger(__name__)


class AlertRouter:
    """Routes alerts to appropriate notification channels based on rules and preferences."""
    
    def __init__(self):
        self.rate_limit_window = 300  # 5 minutes
        self.max_notifications_per_window = 5
    
    def route_alert(self, alert: Alert) -> List[Dict]:
        """
        Route an alert to appropriate notification channels.
        
        Args:
            alert: The alert to route
            
        Returns:
            List of notification task data dictionaries
        """
        notification_tasks = []
        
        try:
            # Get routing rules for this alert
            routing_rules = self._get_routing_rules(alert)
            
            for rule in routing_rules:
                # Get channels for this rule
                channels = self._get_channels_for_rule(alert, rule)
                
                for channel_config in channels:
                    # Get recipients for this channel
                    recipients = self._get_recipients(alert, channel_config)
                    
                    for recipient in recipients:
                        # Check rate limits
                        if not self._check_rate_limit(alert, channel_config['type'], recipient):
                            logger.warning(
                                f"Rate limit exceeded for {channel_config['type']} "
                                f"to {recipient} for alert {alert.id}"
                            )
                            continue
                        
                        # Create notification task
                        task_data = {
                            'alert_id': str(alert.id),
                            'channel_type': channel_config['type'],
                            'recipient': recipient,
                            'priority': self._get_priority_for_severity(alert.severity),
                            'template_id': channel_config.get('template_id'),
                            'config': channel_config.get('config', {}),
                            'delay': channel_config.get('delay', 0)
                        }
                        
                        notification_tasks.append(task_data)
            
            logger.info(
                f"Routed alert {alert.id} to {len(notification_tasks)} notification tasks"
            )
            
            return notification_tasks
            
        except Exception as e:
            logger.error(f"Error routing alert {alert.id}: {e}")
            # Return fallback notification to alert owner
            return self._get_fallback_notifications(alert)
    
    def _get_routing_rules(self, alert: Alert) -> List[Dict]:
        """Get applicable routing rules for an alert."""
        rules = []
        
        # Primary rule from alert rule
        if alert.rule_triggered:
            rule_config = {
                'source': 'alert_rule',
                'rule_id': alert.rule_triggered.id,
                'channels': alert.rule_triggered.notification_channels,
                'priority': alert.rule_triggered.priority
            }
            rules.append(rule_config)
        
        # Default rules based on severity
        default_rule = self._get_default_rule_for_severity(alert.severity)
        if default_rule:
            rules.append(default_rule)
        
        # User preference rules
        user_rules = self._get_user_preference_rules(alert.owner, alert.severity)
        rules.extend(user_rules)
        
        return rules
    
    def _get_channels_for_rule(self, alert: Alert, rule: Dict) -> List[Dict]:
        """Get notification channels for a routing rule."""
        channels = []
        
        if rule['source'] == 'alert_rule':
            # Use channels from alert rule
            for channel_config in rule['channels']:
                if isinstance(channel_config, str):
                    # Simple channel type
                    channels.append({
                        'type': channel_config,
                        'config': {}
                    })
                elif isinstance(channel_config, dict):
                    # Detailed channel configuration
                    channels.append(channel_config)
        
        elif rule['source'] == 'default':
            # Use default channels for severity
            default_channels = self._get_default_channels_for_severity(alert.severity)
            channels.extend(default_channels)
        
        elif rule['source'] == 'user_preference':
            # Use user preference channels
            channels.extend(rule['channels'])
        
        return channels
    
    def _get_recipients(self, alert: Alert, channel_config: Dict) -> List[str]:
        """Get recipients for a notification channel."""
        recipients = []
        
        # Primary recipient is alert owner
        owner_contact = self._get_user_contact_for_channel(
            alert.owner, 
            channel_config['type']
        )
        if owner_contact:
            recipients.append(owner_contact)
        
        # Additional recipients from channel config
        if 'recipients' in channel_config:
            for recipient_config in channel_config['recipients']:
                if isinstance(recipient_config, str):
                    # Direct contact (email, phone, etc.)
                    recipients.append(recipient_config)
                elif isinstance(recipient_config, dict):
                    # User reference or group
                    if 'user_id' in recipient_config:
                        user = User.objects.filter(id=recipient_config['user_id']).first()
                        if user:
                            contact = self._get_user_contact_for_channel(
                                user, 
                                channel_config['type']
                            )
                            if contact:
                                recipients.append(contact)
                    elif 'group' in recipient_config:
                        # Handle group recipients (could be implemented later)
                        pass
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recipients = []
        for recipient in recipients:
            if recipient not in seen:
                seen.add(recipient)
                unique_recipients.append(recipient)
        
        return unique_recipients
    
    def _get_user_contact_for_channel(self, user: User, channel_type: str) -> Optional[str]:
        """Get user contact information for a specific channel type."""
        if channel_type == 'email':
            return user.email
        elif channel_type == 'sms':
            # Assuming user model has phone field
            return getattr(user, 'phone', None)
        elif channel_type == 'slack':
            # Assuming user model has slack_user_id field
            return getattr(user, 'slack_user_id', None)
        elif channel_type == 'teams':
            # Assuming user model has teams_user_id field
            return getattr(user, 'teams_user_id', None)
        elif channel_type == 'webhook':
            # For webhooks, we might use a configured endpoint
            return channel_type  # Will be handled by webhook channel
        
        return None
    
    def _check_rate_limit(self, alert: Alert, channel_type: str, recipient: str) -> bool:
        """Check if notification should be rate limited."""
        # Create rate limit key
        rate_limit_key = f"notification_rate_limit:{channel_type}:{recipient}:{alert.alert_type}"
        
        # Get current count
        current_count = cache.get(rate_limit_key, 0)
        
        if current_count >= self.max_notifications_per_window:
            return False
        
        # Increment counter
        cache.set(rate_limit_key, current_count + 1, self.rate_limit_window)
        
        return True
    
    def _get_priority_for_severity(self, severity: str) -> str:
        """Map alert severity to task priority."""
        priority_mapping = {
            'critical': 'high',
            'high': 'high',
            'medium': 'medium',
            'low': 'low'
        }
        return priority_mapping.get(severity, 'low')
    
    def _get_default_rule_for_severity(self, severity: str) -> Optional[Dict]:
        """Get default routing rule for alert severity."""
        if severity in ['critical', 'high']:
            return {
                'source': 'default',
                'channels': [
                    {'type': 'email', 'config': {}},
                    {'type': 'slack', 'config': {}} if severity == 'critical' else None
                ],
                'priority': 1
            }
        elif severity == 'medium':
            return {
                'source': 'default',
                'channels': [{'type': 'email', 'config': {}}],
                'priority': 2
            }
        
        return None
    
    def _get_default_channels_for_severity(self, severity: str) -> List[Dict]:
        """Get default notification channels for alert severity."""
        if severity == 'critical':
            return [
                {'type': 'email', 'config': {}},
                {'type': 'slack', 'config': {}},
                {'type': 'sms', 'config': {}}
            ]
        elif severity == 'high':
            return [
                {'type': 'email', 'config': {}},
                {'type': 'slack', 'config': {}}
            ]
        elif severity == 'medium':
            return [
                {'type': 'email', 'config': {}}
            ]
        else:  # low
            return [
                {'type': 'email', 'config': {'delay': 300}}  # 5 minute delay
            ]
    
    def _get_user_preference_rules(self, user: User, severity: str) -> List[Dict]:
        """Get user-specific notification preferences."""
        # This could be implemented to read from user preferences model
        # For now, return empty list
        return []
    
    def _get_fallback_notifications(self, alert: Alert) -> List[Dict]:
        """Get fallback notifications when routing fails."""
        fallback_tasks = []
        
        # Always try to notify alert owner via email
        if alert.owner.email:
            fallback_tasks.append({
                'alert_id': str(alert.id),
                'channel_type': 'email',
                'recipient': alert.owner.email,
                'priority': 'high',
                'template_id': None,
                'config': {'fallback': True},
                'delay': 0
            })
        
        return fallback_tasks
    
    def check_channel_failover(self, alert: Alert, failed_channel: str) -> List[Dict]:
        """Handle channel failover when primary channels fail."""
        failover_tasks = []
        
        # Define failover mapping
        failover_mapping = {
            'slack': ['email', 'sms'],
            'teams': ['email', 'slack'],
            'sms': ['email'],
            'webhook': ['email']
        }
        
        failover_channels = failover_mapping.get(failed_channel, ['email'])
        
        for channel_type in failover_channels:
            recipient = self._get_user_contact_for_channel(alert.owner, channel_type)
            if recipient:
                failover_tasks.append({
                    'alert_id': str(alert.id),
                    'channel_type': channel_type,
                    'recipient': recipient,
                    'priority': 'high',
                    'template_id': None,
                    'config': {'failover': True, 'original_channel': failed_channel},
                    'delay': 0
                })
        
        return failover_tasks
    
    def get_escalation_recipients(self, alert: Alert, escalation_level: int) -> List[Dict]:
        """Get recipients for alert escalation."""
        escalation_tasks = []
        
        # This would typically read from escalation rules
        # For now, implement basic escalation logic
        
        if escalation_level == 1:
            # Escalate to team leads
            # This could query a team/role model
            pass
        elif escalation_level == 2:
            # Escalate to managers
            pass
        elif escalation_level >= 3:
            # Escalate to executives
            pass
        
        return escalation_tasks