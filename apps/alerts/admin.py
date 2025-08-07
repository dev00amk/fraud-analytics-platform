from django.contrib import admin
from django.utils.html import format_html
from .models import (
    Alert, AlertRule, NotificationDelivery, 
    EscalationRule, NotificationTemplate, EscalationTask
)


@admin.register(Alert)
class AlertAdmin(admin.ModelAdmin):
    list_display = [
        'alert_type', 'severity', 'status', 'fraud_score', 
        'transaction_link', 'owner', 'created_at'
    ]
    list_filter = ['severity', 'status', 'alert_type', 'created_at']
    search_fields = ['alert_type', 'title', 'transaction__transaction_id']
    readonly_fields = ['id', 'created_at', 'acknowledged_at', 'resolved_at']
    raw_id_fields = ['transaction', 'rule_triggered', 'acknowledged_by']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('id', 'alert_type', 'severity', 'status', 'title', 'message')
        }),
        ('Context', {
            'fields': ('transaction', 'fraud_score', 'risk_factors', 'context_data')
        }),
        ('Metadata', {
            'fields': ('rule_triggered', 'owner')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'acknowledged_at', 'acknowledged_by', 'resolved_at')
        }),
    )

    def transaction_link(self, obj):
        if obj.transaction:
            return format_html(
                '<a href="/admin/transactions/transaction/{}/change/">{}</a>',
                obj.transaction.id,
                obj.transaction.transaction_id
            )
        return '-'
    transaction_link.short_description = 'Transaction'


@admin.register(AlertRule)
class AlertRuleAdmin(admin.ModelAdmin):
    list_display = [
        'name', 'alert_type', 'severity', 'action', 
        'is_active', 'priority', 'owner', 'created_at'
    ]
    list_filter = ['severity', 'action', 'is_active', 'created_at']
    search_fields = ['name', 'alert_type', 'description']
    readonly_fields = ['id', 'created_at', 'updated_at']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('id', 'name', 'description', 'is_active', 'priority')
        }),
        ('Rule Configuration', {
            'fields': ('conditions', 'fraud_score_threshold', 'amount_threshold')
        }),
        ('Alert Settings', {
            'fields': ('alert_type', 'severity', 'action')
        }),
        ('Notification Settings', {
            'fields': ('notification_channels', 'consolidation_window')
        }),
        ('Metadata', {
            'fields': ('owner', 'created_at', 'updated_at')
        }),
    )


@admin.register(NotificationDelivery)
class NotificationDeliveryAdmin(admin.ModelAdmin):
    list_display = [
        'alert_link', 'channel_type', 'recipient', 'status', 
        'attempts', 'last_attempt_at', 'delivered_at'
    ]
    list_filter = ['channel_type', 'status', 'created_at']
    search_fields = ['recipient', 'alert__alert_type']
    readonly_fields = ['id', 'created_at', 'updated_at']
    raw_id_fields = ['alert']

    def alert_link(self, obj):
        if obj.alert:
            return format_html(
                '<a href="/admin/alerts/alert/{}/change/">{}</a>',
                obj.alert.id,
                obj.alert.alert_type
            )
        return '-'
    alert_link.short_description = 'Alert'


@admin.register(EscalationRule)
class EscalationRuleAdmin(admin.ModelAdmin):
    list_display = [
        'name', 'alert_severity', 'timeout_minutes', 
        'business_hours_only', 'is_active', 'owner'
    ]
    list_filter = ['alert_severity', 'business_hours_only', 'is_active']
    search_fields = ['name']
    readonly_fields = ['id', 'created_at', 'updated_at']


@admin.register(NotificationTemplate)
class NotificationTemplateAdmin(admin.ModelAdmin):
    list_display = [
        'name', 'channel_type', 'alert_type', 
        'is_default', 'owner', 'created_at'
    ]
    list_filter = ['channel_type', 'alert_type', 'is_default']
    search_fields = ['name', 'alert_type']
    readonly_fields = ['id', 'created_at', 'updated_at']


@admin.register(EscalationTask)
class EscalationTaskAdmin(admin.ModelAdmin):
    list_display = [
        'alert_link', 'escalation_level', 'status', 
        'scheduled_at', 'executed_at', 'cancelled_at'
    ]
    list_filter = ['status', 'escalation_level', 'scheduled_at']
    readonly_fields = ['id', 'created_at', 'updated_at']
    raw_id_fields = ['alert', 'escalation_rule']

    def alert_link(self, obj):
        if obj.alert:
            return format_html(
                '<a href="/admin/alerts/alert/{}/change/">{}</a>',
                obj.alert.id,
                obj.alert.alert_type
            )
        return '-'
    alert_link.short_description = 'Alert'