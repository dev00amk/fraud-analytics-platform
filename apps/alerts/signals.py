from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Alert, EscalationTask


@receiver(post_save, sender=Alert)
def handle_alert_created(sender, instance, created, **kwargs):
    """Handle alert creation - placeholder for future alert processing."""
    if created:
        # This will be expanded in later tasks to:
        # 1. Trigger notification delivery
        # 2. Schedule escalation tasks
        # 3. Update metrics
        pass


@receiver(post_save, sender=Alert)
def handle_alert_acknowledged(sender, instance, created, **kwargs):
    """Handle alert acknowledgment - cancel escalation tasks."""
    if not created and instance.status == 'acknowledged':
        # Cancel any pending escalation tasks for this alert
        EscalationTask.objects.filter(
            alert=instance,
            status='scheduled'
        ).update(
            status='cancelled',
            cancelled_at=instance.acknowledged_at
        )