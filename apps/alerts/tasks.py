"""
Celery tasks for alert processing with priority queues, retry logic, and rate limiting.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from celery import Celery, Task
from celery.exceptions import Retry
from django.conf import settings
from django.core.cache import cache
from django.db import transaction
from django.utils import timezone

from .models import Alert, NotificationDelivery, EscalationTask
from .services import AlertGenerator

# Get the Celery app instance
from fraud_platform.celery import app

logger = logging.getLogger(__name__)


class AlertProcessingTask(Task):
    """Base task class for alert processing with retry logic and error handling."""
    
    autoretry_for = (Exception,)
    retry_kwargs = {'max_retries': 3, 'countdown': 60}
    retry_backoff = True
    retry_backoff_max = 3600  # 1 hour max delay
    retry_jitter = True
    
    def retry(self, args=None, kwargs=None, exc=None, throw=True, eta=None, countdown=None, max_retries=None, **options):
        """Custom retry logic with exponential backoff."""
        if countdown is None and eta is None:
            # Calculate exponential backoff
            attempt = self.request.retries + 1
            base_delay = 60  # 1 minute base delay
            max_delay = 3600  # 1 hour max delay
            backoff_multiplier = 2
            
            delay = min(base_delay * (backoff_multiplier ** attempt), max_delay)
            countdown = delay
            
        logger.warning(
            f"Retrying task {self.name} (attempt {self.request.retries + 1}): {exc}. "
            f"Next retry in {countdown} seconds."
        )
        
        return super().retry(
            args=args, 
            kwargs=kwargs, 
            exc=exc, 
            throw=throw, 
            eta=eta, 
            countdown=countdown, 
            max_retries=max_retries, 
            **options
        )


@app.task(
    bind=True,
    base=AlertProcessingTask,
    queue='alerts_high_priority',
    routing_key='alerts.high_priority'
)
def process_critical_alert(self, alert_id: str) -> Dict:
    """Process critical alerts with highest priority."""
    return _process_alert_with_priority(self, alert_id, 'critical')


@app.task(
    bind=True,
    base=AlertProcessingTask,
    queue='alerts_high_priority',
    routing_key='alerts.high_priority'
)
def process_high_priority_alert(self, alert_id: str) -> Dict:
    """Process high priority alerts."""
    return _process_alert_with_priority(self, alert_id, 'high')


@app.task(
    bind=True,
    base=AlertProcessingTask,
    queue='alerts_medium_priority',
    routing_key='alerts.medium_priority'
)
def process_medium_priority_alert(self, alert_id: str) -> Dict:
    """Process medium priority alerts."""
    return _process_alert_with_priority(self, alert_id, 'medium')


@app.task(
    bind=True,
    base=AlertProcessingTask,
    queue='alerts_low_priority',
    routing_key='alerts.low_priority'
)
def process_low_priority_alert(self, alert_id: str) -> Dict:
    """Process low priority alerts."""
    return _process_alert_with_priority(self, alert_id, 'low')


def _process_alert_with_priority(task_instance, alert_id: str, expected_priority: str) -> Dict:
    """Internal function to process alerts with priority validation."""
    start_time = time.time()
    
    try:
        with transaction.atomic():
            alert = Alert.objects.select_for_update().get(id=alert_id)
            
            # Validate priority matches expected
            if alert.severity != expected_priority:
                logger.warning(
                    f"Alert {alert_id} priority mismatch. Expected: {expected_priority}, "
                    f"Actual: {alert.severity}"
                )
            
            # Check rate limiting
            if not _check_rate_limit(alert):
                logger.info(f"Alert {alert_id} rate limited, skipping processing")
                return {
                    'alert_id': alert_id,
                    'status': 'rate_limited',
                    'processing_time': time.time() - start_time
                }
            
            # Process the alert
            result = _process_single_alert(alert)
            
            # Update alert status
            alert.status = 'sent' if result['success'] else 'pending'
            alert.save(update_fields=['status'])
            
            processing_time = time.time() - start_time
            logger.info(
                f"Alert {alert_id} processed successfully in {processing_time:.2f}s. "
                f"Notifications sent: {result['notifications_sent']}"
            )
            
            return {
                'alert_id': alert_id,
                'status': 'processed',
                'notifications_sent': result['notifications_sent'],
                'processing_time': processing_time,
                'success': result['success']
            }
            
    except Alert.DoesNotExist:
        logger.error(f"Alert {alert_id} not found")
        return {
            'alert_id': alert_id,
            'status': 'not_found',
            'processing_time': time.time() - start_time,
            'error': 'Alert not found'
        }
    except Exception as exc:
        logger.error(f"Error processing alert {alert_id}: {exc}")
        # Re-raise to trigger retry mechanism
        raise task_instance.retry(exc=exc)


def _process_single_alert(alert: Alert) -> Dict:
    """Process a single alert and send notifications."""
    from .channels.router import AlertRouter
    
    router = AlertRouter()
    notifications_sent = 0
    failed_notifications = 0
    
    try:
        # Route alert to appropriate channels
        notification_tasks = router.route_alert(alert)
        
        for task_data in notification_tasks:
            try:
                # Create notification delivery record
                delivery = NotificationDelivery.objects.create(
                    alert=alert,
                    channel_type=task_data['channel_type'],
                    recipient=task_data['recipient'],
                    status='pending'
                )
                
                # Queue notification delivery
                send_notification_task.apply_async(
                    args=[str(delivery.id)],
                    queue=f"notifications_{task_data['priority']}",
                    routing_key=f"notifications.{task_data['priority']}"
                )
                
                notifications_sent += 1
                
            except Exception as e:
                logger.error(f"Failed to queue notification for alert {alert.id}: {e}")
                failed_notifications += 1
        
        # Schedule escalation if configured
        if alert.rule_triggered and hasattr(alert.rule_triggered, 'escalation_rules'):
            schedule_escalation_task.apply_async(
                args=[str(alert.id)],
                queue='escalations',
                routing_key='escalations.schedule'
            )
        
        return {
            'success': failed_notifications == 0,
            'notifications_sent': notifications_sent,
            'failed_notifications': failed_notifications
        }
        
    except Exception as e:
        logger.error(f"Error processing alert {alert.id}: {e}")
        return {
            'success': False,
            'notifications_sent': notifications_sent,
            'failed_notifications': failed_notifications + 1,
            'error': str(e)
        }


@app.task(
    bind=True,
    base=AlertProcessingTask,
    queue='notifications_high',
    routing_key='notifications.high'
)
def send_notification_task(self, delivery_id: str) -> Dict:
    """Send individual notification with retry logic."""
    start_time = time.time()
    
    try:
        with transaction.atomic():
            delivery = NotificationDelivery.objects.select_for_update().get(id=delivery_id)
            
            # Check if already processed
            if delivery.status in ['delivered', 'sent']:
                return {
                    'delivery_id': delivery_id,
                    'status': 'already_processed',
                    'processing_time': time.time() - start_time
                }
            
            # Update attempt count
            delivery.attempts += 1
            delivery.last_attempt_at = timezone.now()
            delivery.status = 'retrying' if delivery.attempts > 1 else 'pending'
            delivery.save(update_fields=['attempts', 'last_attempt_at', 'status'])
            
            # Get appropriate channel
            from .channels.factory import NotificationChannelFactory
            
            channel = NotificationChannelFactory.get_channel(delivery.channel_type)
            if not channel:
                raise ValueError(f"Unknown channel type: {delivery.channel_type}")
            
            # Send notification
            result = channel.send_notification(delivery)
            
            # Update delivery status based on result
            if result.success:
                delivery.status = 'delivered' if result.delivered else 'sent'
                delivery.delivered_at = timezone.now() if result.delivered else None
                delivery.external_id = result.external_id or ''
                delivery.error_message = ''
            else:
                delivery.status = 'failed'
                delivery.error_message = result.error_message or 'Unknown error'
                
                # Determine if we should retry
                if delivery.attempts < 3 and result.should_retry:
                    # Schedule retry with exponential backoff
                    retry_delay = min(60 * (2 ** delivery.attempts), 3600)  # Max 1 hour
                    delivery.retry_after = timezone.now() + timedelta(seconds=retry_delay)
                    delivery.status = 'retrying'
                    
                    # Schedule retry
                    self.apply_async(
                        args=[delivery_id],
                        countdown=retry_delay,
                        queue=f'notifications_{delivery.alert.severity}',
                        routing_key=f'notifications.{delivery.alert.severity}'
                    )
            
            delivery.save(update_fields=[
                'status', 'delivered_at', 'external_id', 'error_message', 'retry_after'
            ])
            
            processing_time = time.time() - start_time
            logger.info(
                f"Notification {delivery_id} processed. Status: {delivery.status}. "
                f"Time: {processing_time:.2f}s"
            )
            
            return {
                'delivery_id': delivery_id,
                'status': delivery.status,
                'processing_time': processing_time,
                'attempts': delivery.attempts,
                'success': result.success if 'result' in locals() else False
            }
            
    except NotificationDelivery.DoesNotExist:
        logger.error(f"Notification delivery {delivery_id} not found")
        return {
            'delivery_id': delivery_id,
            'status': 'not_found',
            'processing_time': time.time() - start_time,
            'error': 'Delivery record not found'
        }
    except Exception as exc:
        logger.error(f"Error sending notification {delivery_id}: {exc}")
        # Re-raise to trigger retry mechanism
        raise self.retry(exc=exc)


@app.task(
    bind=True,
    base=AlertProcessingTask,
    queue='escalations',
    routing_key='escalations.schedule'
)
def schedule_escalation_task(self, alert_id: str) -> Dict:
    """Schedule escalation tasks for an alert."""
    try:
        alert = Alert.objects.get(id=alert_id)
        
        # Find applicable escalation rules
        from .models import EscalationRule
        escalation_rules = EscalationRule.objects.filter(
            is_active=True,
            alert_severity=alert.severity
        )
        
        scheduled_tasks = 0
        
        for rule in escalation_rules:
            # Calculate escalation time
            escalation_time = timezone.now() + timedelta(minutes=rule.timeout_minutes)
            
            # Create escalation task
            escalation_task = EscalationTask.objects.create(
                alert=alert,
                escalation_rule=rule,
                escalation_level=1,  # Start with level 1
                scheduled_at=escalation_time,
                status='scheduled'
            )
            
            # Schedule the escalation
            execute_escalation_task.apply_async(
                args=[str(escalation_task.id)],
                eta=escalation_time,
                queue='escalations',
                routing_key='escalations.execute'
            )
            
            scheduled_tasks += 1
            
        logger.info(f"Scheduled {scheduled_tasks} escalation tasks for alert {alert_id}")
        
        return {
            'alert_id': alert_id,
            'scheduled_tasks': scheduled_tasks,
            'status': 'success'
        }
        
    except Alert.DoesNotExist:
        logger.error(f"Alert {alert_id} not found for escalation scheduling")
        return {
            'alert_id': alert_id,
            'status': 'not_found',
            'error': 'Alert not found'
        }
    except Exception as exc:
        logger.error(f"Error scheduling escalation for alert {alert_id}: {exc}")
        raise self.retry(exc=exc)


@app.task(
    bind=True,
    base=AlertProcessingTask,
    queue='escalations',
    routing_key='escalations.execute'
)
def execute_escalation_task(self, escalation_task_id: str) -> Dict:
    """Execute an escalation task."""
    try:
        with transaction.atomic():
            escalation_task = EscalationTask.objects.select_for_update().get(
                id=escalation_task_id
            )
            
            # Check if escalation is still needed
            alert = escalation_task.alert
            if alert.status in ['acknowledged', 'resolved']:
                escalation_task.status = 'cancelled'
                escalation_task.cancelled_at = timezone.now()
                escalation_task.save(update_fields=['status', 'cancelled_at'])
                
                return {
                    'escalation_task_id': escalation_task_id,
                    'status': 'cancelled',
                    'reason': f'Alert already {alert.status}'
                }
            
            # Execute escalation
            from .services import EscalationEngine
            
            engine = EscalationEngine()
            result = engine.execute_escalation(escalation_task)
            
            # Update escalation task
            escalation_task.status = 'executed'
            escalation_task.executed_at = timezone.now()
            escalation_task.save(update_fields=['status', 'executed_at'])
            
            # Schedule next level if needed
            if result.get('schedule_next_level'):
                next_level = escalation_task.escalation_level + 1
                next_escalation_time = timezone.now() + timedelta(
                    minutes=escalation_task.escalation_rule.timeout_minutes
                )
                
                next_task = EscalationTask.objects.create(
                    alert=alert,
                    escalation_rule=escalation_task.escalation_rule,
                    escalation_level=next_level,
                    scheduled_at=next_escalation_time,
                    status='scheduled'
                )
                
                execute_escalation_task.apply_async(
                    args=[str(next_task.id)],
                    eta=next_escalation_time,
                    queue='escalations',
                    routing_key='escalations.execute'
                )
            
            return {
                'escalation_task_id': escalation_task_id,
                'status': 'executed',
                'notifications_sent': result.get('notifications_sent', 0),
                'next_level_scheduled': result.get('schedule_next_level', False)
            }
            
    except EscalationTask.DoesNotExist:
        logger.error(f"Escalation task {escalation_task_id} not found")
        return {
            'escalation_task_id': escalation_task_id,
            'status': 'not_found',
            'error': 'Escalation task not found'
        }
    except Exception as exc:
        logger.error(f"Error executing escalation task {escalation_task_id}: {exc}")
        raise self.retry(exc=exc)


@app.task(queue='maintenance', routing_key='maintenance.cleanup')
def cleanup_failed_deliveries() -> Dict:
    """Clean up old failed delivery records and move to dead letter queue."""
    cutoff_time = timezone.now() - timedelta(days=7)  # 7 days old
    
    # Find persistently failed deliveries
    failed_deliveries = NotificationDelivery.objects.filter(
        status='failed',
        attempts__gte=3,
        last_attempt_at__lt=cutoff_time
    )
    
    dead_letter_count = 0
    
    for delivery in failed_deliveries:
        # Log to dead letter queue (could be a separate model or external system)
        logger.error(
            f"Moving delivery {delivery.id} to dead letter queue. "
            f"Alert: {delivery.alert.id}, Channel: {delivery.channel_type}, "
            f"Recipient: {delivery.recipient}, Error: {delivery.error_message}"
        )
        
        # Mark as dead letter
        delivery.status = 'dead_letter'
        delivery.save(update_fields=['status'])
        
        dead_letter_count += 1
    
    # Clean up old successful deliveries (optional)
    old_successful = NotificationDelivery.objects.filter(
        status__in=['delivered', 'sent'],
        created_at__lt=timezone.now() - timedelta(days=30)
    )
    
    deleted_count = old_successful.count()
    old_successful.delete()
    
    logger.info(
        f"Cleanup completed. Dead letter: {dead_letter_count}, "
        f"Deleted old records: {deleted_count}"
    )
    
    return {
        'dead_letter_count': dead_letter_count,
        'deleted_count': deleted_count,
        'status': 'success'
    }


def _check_rate_limit(alert: Alert) -> bool:
    """Check if alert processing should be rate limited."""
    # Rate limiting key based on alert type and recipient
    rate_limit_key = f"alert_rate_limit:{alert.alert_type}:{alert.owner.id}"
    
    # Get current count
    current_count = cache.get(rate_limit_key, 0)
    
    # Rate limit: max 10 alerts per minute per user per alert type
    max_alerts_per_minute = 10
    
    if current_count >= max_alerts_per_minute:
        return False
    
    # Increment counter with 60 second expiry
    cache.set(rate_limit_key, current_count + 1, 60)
    
    return True


# Task routing configuration
def route_alert_task(alert_severity: str) -> str:
    """Route alert to appropriate task based on severity."""
    task_mapping = {
        'critical': 'apps.alerts.tasks.process_critical_alert',
        'high': 'apps.alerts.tasks.process_high_priority_alert',
        'medium': 'apps.alerts.tasks.process_medium_priority_alert',
        'low': 'apps.alerts.tasks.process_low_priority_alert',
    }
    
    return task_mapping.get(alert_severity, 'apps.alerts.tasks.process_low_priority_alert')


# Periodic tasks for maintenance
@app.task(queue='maintenance', routing_key='maintenance.health_check')
def health_check_task() -> Dict:
    """Perform health checks on the alert processing system."""
    from django.db import connection
    
    health_status = {
        'database': False,
        'cache': False,
        'queues': {},
        'timestamp': timezone.now().isoformat()
    }
    
    # Check database connectivity
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        health_status['database'] = True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
    
    # Check cache connectivity
    try:
        cache.set('health_check', 'ok', 10)
        health_status['cache'] = cache.get('health_check') == 'ok'
    except Exception as e:
        logger.error(f"Cache health check failed: {e}")
    
    # Check queue lengths (simplified)
    try:
        # This would need to be implemented based on your broker
        # For Redis, you could check queue lengths
        health_status['queues'] = {
            'alerts_high_priority': 'unknown',
            'alerts_medium_priority': 'unknown',
            'alerts_low_priority': 'unknown',
            'notifications_high': 'unknown',
            'escalations': 'unknown'
        }
    except Exception as e:
        logger.error(f"Queue health check failed: {e}")
    
    return health_status