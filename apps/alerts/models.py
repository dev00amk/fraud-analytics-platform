import uuid
from datetime import timedelta
from django.contrib.auth import get_user_model
from django.db import models
from apps.transactions.models import Transaction

User = get_user_model()


class Alert(models.Model):
    """Core alert model for fraud notifications."""
    
    SEVERITY_CHOICES = [
        ('low', 'Low'),
        ('medium', 'Medium'), 
        ('high', 'High'),
        ('critical', 'Critical'),
    ]
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('sent', 'Sent'),
        ('acknowledged', 'Acknowledged'),
        ('resolved', 'Resolved'),
        ('escalated', 'Escalated'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    alert_type = models.CharField(max_length=100)
    severity = models.CharField(max_length=20, choices=SEVERITY_CHOICES)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    
    # Context
    transaction = models.ForeignKey(Transaction, on_delete=models.CASCADE, related_name='alerts')
    fraud_score = models.FloatField()
    risk_factors = models.JSONField(default=dict)
    
    # Content
    title = models.CharField(max_length=255)
    message = models.TextField()
    context_data = models.JSONField(default=dict)
    
    # Metadata
    rule_triggered = models.ForeignKey('AlertRule', on_delete=models.SET_NULL, null=True, blank=True)
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name='alerts')
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    acknowledged_at = models.DateTimeField(null=True, blank=True)
    acknowledged_by = models.ForeignKey(
        User, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True, 
        related_name='acknowledged_alerts'
    )
    resolved_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['status', 'severity']),
            models.Index(fields=['created_at']),
            models.Index(fields=['owner', 'status']),
            models.Index(fields=['transaction']),
        ]

    def __str__(self):
        return f"Alert: {self.alert_type} - {self.transaction.transaction_id}"


class AlertRule(models.Model):
    """Configurable rules for alert generation."""
    
    ACTION_CHOICES = [
        ('alert', 'Generate Alert'),
        ('escalate', 'Immediate Escalation'),
        ('block', 'Block and Alert'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    
    # Rule conditions
    conditions = models.JSONField(default=dict)  # Complex rule conditions
    fraud_score_threshold = models.FloatField(null=True, blank=True)
    amount_threshold = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    
    # Alert configuration
    alert_type = models.CharField(max_length=100)
    severity = models.CharField(max_length=20, choices=Alert.SEVERITY_CHOICES)
    action = models.CharField(max_length=20, choices=ACTION_CHOICES)
    
    # Notification settings
    notification_channels = models.JSONField(default=list)
    consolidation_window = models.DurationField(default=timedelta(minutes=10))
    
    # Metadata
    is_active = models.BooleanField(default=True)
    priority = models.IntegerField(default=1)
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name='alert_rules')
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['priority', 'name']
        indexes = [
            models.Index(fields=['is_active', 'priority']),
            models.Index(fields=['owner']),
        ]

    def __str__(self):
        return f"Rule: {self.name}"


class NotificationDelivery(models.Model):
    """Tracks notification delivery attempts and status."""
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('sent', 'Sent'),
        ('delivered', 'Delivered'),
        ('failed', 'Failed'),
        ('retrying', 'Retrying'),
    ]
    
    CHANNEL_CHOICES = [
        ('email', 'Email'),
        ('sms', 'SMS'),
        ('slack', 'Slack'),
        ('webhook', 'Webhook'),
        ('teams', 'Microsoft Teams'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    alert = models.ForeignKey(Alert, on_delete=models.CASCADE, related_name='deliveries')
    
    # Delivery details
    channel_type = models.CharField(max_length=50, choices=CHANNEL_CHOICES)
    recipient = models.CharField(max_length=255)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    
    # Delivery tracking
    external_id = models.CharField(max_length=255, blank=True)
    attempts = models.IntegerField(default=0)
    last_attempt_at = models.DateTimeField(null=True, blank=True)
    delivered_at = models.DateTimeField(null=True, blank=True)
    
    # Error handling
    error_message = models.TextField(blank=True)
    retry_after = models.DateTimeField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['alert', 'channel_type']),
            models.Index(fields=['status']),
            models.Index(fields=['retry_after']),
        ]

    def __str__(self):
        return f"Delivery: {self.channel_type} to {self.recipient}"


class EscalationRule(models.Model):
    """Rules for alert escalation workflows."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    
    # Escalation conditions
    alert_severity = models.CharField(max_length=20, choices=Alert.SEVERITY_CHOICES)
    timeout_minutes = models.IntegerField()
    business_hours_only = models.BooleanField(default=False)
    
    # Escalation path
    escalation_levels = models.JSONField(default=list)  # List of escalation levels with recipients
    
    # Metadata
    is_active = models.BooleanField(default=True)
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name='escalation_rules')
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['timeout_minutes']
        indexes = [
            models.Index(fields=['is_active', 'alert_severity']),
            models.Index(fields=['owner']),
        ]

    def __str__(self):
        return f"Escalation: {self.name}"


class NotificationTemplate(models.Model):
    """Templates for notification messages."""
    
    CHANNEL_CHOICES = [
        ('email', 'Email'),
        ('sms', 'SMS'),
        ('slack', 'Slack'),
        ('webhook', 'Webhook'),
        ('teams', 'Microsoft Teams'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    channel_type = models.CharField(max_length=20, choices=CHANNEL_CHOICES)
    alert_type = models.CharField(max_length=100)
    
    # Template content
    subject_template = models.CharField(max_length=255, blank=True)
    body_template = models.TextField()
    
    # Template variables and formatting
    variables = models.JSONField(default=dict)
    formatting_options = models.JSONField(default=dict)
    
    # Metadata
    is_default = models.BooleanField(default=False)
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name='notification_templates')
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['channel_type', 'alert_type', 'name']
        indexes = [
            models.Index(fields=['channel_type', 'alert_type']),
            models.Index(fields=['is_default']),
            models.Index(fields=['owner']),
        ]
        unique_together = [['channel_type', 'alert_type', 'is_default', 'owner']]

    def __str__(self):
        return f"Template: {self.name} ({self.channel_type})"


class EscalationTask(models.Model):
    """Tracks scheduled escalation tasks."""
    
    STATUS_CHOICES = [
        ('scheduled', 'Scheduled'),
        ('executed', 'Executed'),
        ('cancelled', 'Cancelled'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    alert = models.ForeignKey(Alert, on_delete=models.CASCADE, related_name='escalation_tasks')
    escalation_rule = models.ForeignKey(EscalationRule, on_delete=models.CASCADE)
    
    # Escalation details
    escalation_level = models.IntegerField()
    scheduled_at = models.DateTimeField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='scheduled')
    
    # Execution tracking
    executed_at = models.DateTimeField(null=True, blank=True)
    cancelled_at = models.DateTimeField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['scheduled_at']
        indexes = [
            models.Index(fields=['status', 'scheduled_at']),
            models.Index(fields=['alert']),
        ]

    def __str__(self):
        return f"Escalation Task: {self.alert.alert_type} - Level {self.escalation_level}"