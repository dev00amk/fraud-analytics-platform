"""
Celery configuration for fraud_platform project.
"""

import os

from celery import Celery

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "fraud_platform.settings")

app = Celery("fraud_platform")

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
app.config_from_object("django.conf:settings", namespace="CELERY")

# Load task modules from all registered Django apps.
app.autodiscover_tasks()

# Configure task routing for alert processing
app.conf.task_routes = {
    # Alert processing tasks
    'apps.alerts.tasks.process_critical_alert': {'queue': 'alerts_high_priority'},
    'apps.alerts.tasks.process_high_priority_alert': {'queue': 'alerts_high_priority'},
    'apps.alerts.tasks.process_medium_priority_alert': {'queue': 'alerts_medium_priority'},
    'apps.alerts.tasks.process_low_priority_alert': {'queue': 'alerts_low_priority'},
    
    # Notification tasks
    'apps.alerts.tasks.send_notification_task': {'queue': 'notifications_high'},
    
    # Escalation tasks
    'apps.alerts.tasks.schedule_escalation_task': {'queue': 'escalations'},
    'apps.alerts.tasks.execute_escalation_task': {'queue': 'escalations'},
    
    # Maintenance tasks
    'apps.alerts.tasks.cleanup_failed_deliveries': {'queue': 'maintenance'},
    'apps.alerts.tasks.health_check_task': {'queue': 'maintenance'},
}

# Configure queue priorities
app.conf.task_queue_max_priority = 10
app.conf.task_default_priority = 5

# Configure worker settings
app.conf.worker_prefetch_multiplier = 1
app.conf.task_acks_late = True
app.conf.worker_disable_rate_limits = False
