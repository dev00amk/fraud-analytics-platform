"""
Escalation engine for managing alert escalation workflows.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from django.contrib.auth import get_user_model
from django.utils import timezone

from ..models import Alert, EscalationRule, EscalationTask, NotificationDelivery

User = get_user_model()
logger = logging.getLogger(__name__)


class EscalationResult:
    """Result of an escalation execution."""
    
    def __init__(self, success: bool = False, notifications_sent: int = 0, 
                 schedule_next_level: bool = False, error_message: str = None):
        self.success = success
        self.notifications_sent = notifications_sent
        self.schedule_next_level = schedule_next_level
        self.error_message = error_message
    
    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'notifications_sent': self.notifications_sent,
            'schedule_next_level': self.schedule_next_level,
            'error_message': self.error_message
        }


class EscalationEngine:
    """Manages alert escalation workflows and execution."""
    
    def __init__(self):
        self.max_escalation_levels = 5
    
    def execute_escalation(self, escalation_task: EscalationTask) -> EscalationResult:
        """
        Execute an escalation task.
        
        Args:
            escalation_task: The escalation task to execute
            
        Returns:
            EscalationResult with execution details
        """
        try:
            alert = escalation_task.alert
            rule = escalation_task.escalation_rule
            level = escalation_task.escalation_level
            
            logger.info(
                f"Executing escalation for alert {alert.id}, "
                f"rule {rule.name}, level {level}"
            )
            
            # Check if escalation is still needed
            if alert.status in ['acknowledged', 'resolved']:
                logger.info(f"Alert {alert.id} already {alert.status}, skipping escalation")
                return EscalationResult(success=True, notifications_sent=0)
            
            # Check business hours if required
            if rule.business_hours_only and not self._is_business_hours():
                logger.info(f"Outside business hours, skipping escalation for alert {alert.id}")
                return EscalationResult(success=True, notifications_sent=0)
            
            # Get escalation level configuration
            level_config = self._get_escalation_level_config(rule, level)
            if not level_config:
                logger.warning(f"No configuration for escalation level {level}")
                return EscalationResult(success=False, error_message="No level configuration")
            
            # Send escalation notifications
            notifications_sent = self._send_escalation_notifications(
                alert, level_config, level
            )
            
            # Update alert status
            alert.status = 'escalated'
            alert.save(update_fields=['status'])
            
            # Determine if next level should be scheduled
            schedule_next = (
                level < self.max_escalation_levels and 
                level < len(rule.escalation_levels) and
                notifications_sent > 0
            )
            
            logger.info(
                f"Escalation executed for alert {alert.id}. "
                f"Notifications sent: {notifications_sent}, "
                f"Schedule next level: {schedule_next}"
            )
            
            return EscalationResult(
                success=True,
                notifications_sent=notifications_sent,
                schedule_next_level=schedule_next
            )
            
        except Exception as e:
            logger.error(f"Error executing escalation task {escalation_task.id}: {e}")
            return EscalationResult(
                success=False,
                error_message=str(e)
            )
    
    def cancel_escalation(self, alert: Alert) -> bool:
        """
        Cancel all pending escalations for an alert.
        
        Args:
            alert: The alert to cancel escalations for
            
        Returns:
            True if escalations were cancelled successfully
        """
        try:
            # Cancel all scheduled escalation tasks
            cancelled_count = EscalationTask.objects.filter(
                alert=alert,
                status='scheduled'
            ).update(
                status='cancelled',
                cancelled_at=timezone.now()
            )
            
            logger.info(f"Cancelled {cancelled_count} escalation tasks for alert {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling escalations for alert {alert.id}: {e}")
            return False
    
    def get_escalation_path(self, alert: Alert) -> List[Dict]:
        """
        Get the escalation path for an alert.
        
        Args:
            alert: The alert to get escalation path for
            
        Returns:
            List of escalation level configurations
        """
        escalation_path = []
        
        try:
            # Find applicable escalation rules
            rules = EscalationRule.objects.filter(
                is_active=True,
                alert_severity=alert.severity
            ).order_by('timeout_minutes')
            
            for rule in rules:
                for level, level_config in enumerate(rule.escalation_levels, 1):
                    escalation_path.append({
                        'rule_id': rule.id,
                        'rule_name': rule.name,
                        'level': level,
                        'timeout_minutes': rule.timeout_minutes,
                        'recipients': level_config.get('recipients', []),
                        'channels': level_config.get('channels', ['email']),
                        'business_hours_only': rule.business_hours_only
                    })
            
            return escalation_path
            
        except Exception as e:
            logger.error(f"Error getting escalation path for alert {alert.id}: {e}")
            return []
    
    def _get_escalation_level_config(self, rule: EscalationRule, level: int) -> Optional[Dict]:
        """Get configuration for a specific escalation level."""
        if level <= 0 or level > len(rule.escalation_levels):
            return None
        
        return rule.escalation_levels[level - 1]  # Convert to 0-based index
    
    def _send_escalation_notifications(self, alert: Alert, level_config: Dict, level: int) -> int:
        """Send notifications for an escalation level."""
        notifications_sent = 0
        
        try:
            recipients = level_config.get('recipients', [])
            channels = level_config.get('channels', ['email'])
            
            for recipient_config in recipients:
                for channel_type in channels:
                    try:
                        # Create notification delivery record
                        delivery = NotificationDelivery.objects.create(
                            alert=alert,
                            channel_type=channel_type,
                            recipient=self._resolve_recipient(recipient_config, channel_type),
                            status='pending'
                        )
                        
                        # Queue notification (import here to avoid circular imports)
                        from ..tasks import send_notification_task
                        
                        send_notification_task.apply_async(
                            args=[str(delivery.id)],
                            queue=f"notifications_{alert.severity}",
                            routing_key=f"notifications.{alert.severity}"
                        )
                        
                        notifications_sent += 1
                        
                    except Exception as e:
                        logger.error(
                            f"Error queuing escalation notification for alert {alert.id}: {e}"
                        )
            
            return notifications_sent
            
        except Exception as e:
            logger.error(f"Error sending escalation notifications: {e}")
            return 0
    
    def _resolve_recipient(self, recipient_config: Dict, channel_type: str) -> str:
        """Resolve recipient configuration to actual contact information."""
        if isinstance(recipient_config, str):
            # Direct contact information
            return recipient_config
        
        if isinstance(recipient_config, dict):
            # User reference
            if 'user_id' in recipient_config:
                try:
                    user = User.objects.get(id=recipient_config['user_id'])
                    return self._get_user_contact_for_channel(user, channel_type)
                except User.DoesNotExist:
                    logger.warning(f"User {recipient_config['user_id']} not found")
                    return None
            
            # Group reference
            elif 'group' in recipient_config:
                # This could be implemented to resolve group members
                logger.warning("Group recipients not yet implemented")
                return None
            
            # Role reference
            elif 'role' in recipient_config:
                # This could be implemented to resolve users by role
                logger.warning("Role-based recipients not yet implemented")
                return None
        
        return None
    
    def _get_user_contact_for_channel(self, user: User, channel_type: str) -> Optional[str]:
        """Get user contact information for a specific channel type."""
        if channel_type == 'email':
            return user.email
        elif channel_type == 'sms':
            return getattr(user, 'phone', None)
        elif channel_type == 'slack':
            return getattr(user, 'slack_user_id', None)
        elif channel_type == 'teams':
            return getattr(user, 'teams_user_id', None)
        
        return None
    
    def _is_business_hours(self) -> bool:
        """Check if current time is within business hours."""
        now = timezone.now()
        
        # Simple business hours check (9 AM - 5 PM, Monday-Friday)
        # This could be made configurable
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        hour = now.hour
        return 9 <= hour < 17
    
    def get_escalation_status(self, alert: Alert) -> Dict:
        """Get current escalation status for an alert."""
        try:
            escalation_tasks = EscalationTask.objects.filter(
                alert=alert
            ).order_by('escalation_level', 'created_at')
            
            status = {
                'alert_id': str(alert.id),
                'has_escalations': escalation_tasks.exists(),
                'escalation_tasks': [],
                'current_level': 0,
                'next_escalation': None
            }
            
            for task in escalation_tasks:
                task_info = {
                    'id': str(task.id),
                    'level': task.escalation_level,
                    'rule_name': task.escalation_rule.name,
                    'status': task.status,
                    'scheduled_at': task.scheduled_at.isoformat() if task.scheduled_at else None,
                    'executed_at': task.executed_at.isoformat() if task.executed_at else None,
                    'cancelled_at': task.cancelled_at.isoformat() if task.cancelled_at else None
                }
                
                status['escalation_tasks'].append(task_info)
                
                if task.status == 'executed':
                    status['current_level'] = max(status['current_level'], task.escalation_level)
                elif task.status == 'scheduled' and not status['next_escalation']:
                    status['next_escalation'] = task_info
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting escalation status for alert {alert.id}: {e}")
            return {
                'alert_id': str(alert.id),
                'has_escalations': False,
                'error': str(e)
            }