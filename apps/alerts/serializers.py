from rest_framework import serializers
from .models import (
    Alert, AlertRule, NotificationDelivery, 
    EscalationRule, NotificationTemplate, EscalationTask
)


class AlertSerializer(serializers.ModelSerializer):
    """Serializer for Alert model."""
    
    transaction_id = serializers.CharField(source='transaction.transaction_id', read_only=True)
    acknowledged_by_username = serializers.CharField(source='acknowledged_by.username', read_only=True)
    
    class Meta:
        model = Alert
        fields = [
            'id', 'alert_type', 'severity', 'status', 'transaction', 'transaction_id',
            'fraud_score', 'risk_factors', 'title', 'message', 'context_data',
            'rule_triggered', 'owner', 'created_at', 'acknowledged_at', 
            'acknowledged_by', 'acknowledged_by_username', 'resolved_at'
        ]
        read_only_fields = ['id', 'created_at', 'acknowledged_at', 'resolved_at', 'owner']
    
    def create(self, validated_data):
        validated_data['owner'] = self.context['request'].user
        return super().create(validated_data)


class AlertRuleSerializer(serializers.ModelSerializer):
    """Serializer for AlertRule model."""
    
    class Meta:
        model = AlertRule
        fields = [
            'id', 'name', 'description', 'conditions', 'fraud_score_threshold',
            'amount_threshold', 'alert_type', 'severity', 'action',
            'notification_channels', 'consolidation_window', 'is_active',
            'priority', 'owner', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at', 'owner']
    
    def create(self, validated_data):
        validated_data['owner'] = self.context['request'].user
        return super().create(validated_data)


class NotificationDeliverySerializer(serializers.ModelSerializer):
    """Serializer for NotificationDelivery model."""
    
    alert_type = serializers.CharField(source='alert.alert_type', read_only=True)
    alert_title = serializers.CharField(source='alert.title', read_only=True)
    
    class Meta:
        model = NotificationDelivery
        fields = [
            'id', 'alert', 'alert_type', 'alert_title', 'channel_type',
            'recipient', 'status', 'external_id', 'attempts',
            'last_attempt_at', 'delivered_at', 'error_message',
            'retry_after', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']


class EscalationRuleSerializer(serializers.ModelSerializer):
    """Serializer for EscalationRule model."""
    
    class Meta:
        model = EscalationRule
        fields = [
            'id', 'name', 'alert_severity', 'timeout_minutes',
            'business_hours_only', 'escalation_levels', 'is_active',
            'owner', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at', 'owner']
    
    def create(self, validated_data):
        validated_data['owner'] = self.context['request'].user
        return super().create(validated_data)


class NotificationTemplateSerializer(serializers.ModelSerializer):
    """Serializer for NotificationTemplate model."""
    
    class Meta:
        model = NotificationTemplate
        fields = [
            'id', 'name', 'channel_type', 'alert_type', 'subject_template',
            'body_template', 'variables', 'formatting_options', 'is_default',
            'owner', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at', 'owner']
    
    def create(self, validated_data):
        validated_data['owner'] = self.context['request'].user
        return super().create(validated_data)


class EscalationTaskSerializer(serializers.ModelSerializer):
    """Serializer for EscalationTask model."""
    
    alert_type = serializers.CharField(source='alert.alert_type', read_only=True)
    alert_title = serializers.CharField(source='alert.title', read_only=True)
    escalation_rule_name = serializers.CharField(source='escalation_rule.name', read_only=True)
    
    class Meta:
        model = EscalationTask
        fields = [
            'id', 'alert', 'alert_type', 'alert_title', 'escalation_rule',
            'escalation_rule_name', 'escalation_level', 'scheduled_at',
            'status', 'executed_at', 'cancelled_at', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']