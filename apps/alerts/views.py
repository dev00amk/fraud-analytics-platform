from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from rest_framework import viewsets, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from .models import (
    Alert, AlertRule, NotificationDelivery, 
    EscalationRule, NotificationTemplate, EscalationTask
)
from .serializers import (
    AlertSerializer, AlertRuleSerializer, NotificationDeliverySerializer,
    EscalationRuleSerializer, NotificationTemplateSerializer, EscalationTaskSerializer
)


class AlertViewSet(viewsets.ModelViewSet):
    """ViewSet for managing alerts."""
    serializer_class = AlertSerializer
    permission_classes = [IsAuthenticated]
    queryset = Alert.objects.all()
    
    def get_queryset(self):
        return Alert.objects.filter(owner=self.request.user)


class AlertRuleViewSet(viewsets.ModelViewSet):
    """ViewSet for managing alert rules."""
    serializer_class = AlertRuleSerializer
    permission_classes = [IsAuthenticated]
    queryset = AlertRule.objects.all()
    
    def get_queryset(self):
        return AlertRule.objects.filter(owner=self.request.user)


class NotificationDeliveryViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for viewing notification deliveries."""
    serializer_class = NotificationDeliverySerializer
    permission_classes = [IsAuthenticated]
    queryset = NotificationDelivery.objects.all()
    
    def get_queryset(self):
        return NotificationDelivery.objects.filter(alert__owner=self.request.user)


class EscalationRuleViewSet(viewsets.ModelViewSet):
    """ViewSet for managing escalation rules."""
    serializer_class = EscalationRuleSerializer
    permission_classes = [IsAuthenticated]
    queryset = EscalationRule.objects.all()
    
    def get_queryset(self):
        return EscalationRule.objects.filter(owner=self.request.user)


class NotificationTemplateViewSet(viewsets.ModelViewSet):
    """ViewSet for managing notification templates."""
    serializer_class = NotificationTemplateSerializer
    permission_classes = [IsAuthenticated]
    queryset = NotificationTemplate.objects.all()
    
    def get_queryset(self):
        return NotificationTemplate.objects.filter(owner=self.request.user)


class EscalationTaskViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for viewing escalation tasks."""
    serializer_class = EscalationTaskSerializer
    permission_classes = [IsAuthenticated]
    queryset = EscalationTask.objects.all()
    
    def get_queryset(self):
        return EscalationTask.objects.filter(alert__owner=self.request.user)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def acknowledge_alert(request, alert_id):
    """Acknowledge an alert."""
    alert = get_object_or_404(Alert, id=alert_id, owner=request.user)
    
    if alert.status == 'acknowledged':
        return Response(
            {'error': 'Alert is already acknowledged'}, 
            status=status.HTTP_400_BAD_REQUEST
        )
    
    alert.status = 'acknowledged'
    alert.acknowledged_at = timezone.now()
    alert.acknowledged_by = request.user
    alert.save()
    
    return Response({
        'message': 'Alert acknowledged successfully',
        'alert_id': str(alert.id),
        'acknowledged_at': alert.acknowledged_at.isoformat()
    })


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def resolve_alert(request, alert_id):
    """Resolve an alert."""
    alert = get_object_or_404(Alert, id=alert_id, owner=request.user)
    
    if alert.status == 'resolved':
        return Response(
            {'error': 'Alert is already resolved'}, 
            status=status.HTTP_400_BAD_REQUEST
        )
    
    alert.status = 'resolved'
    alert.resolved_at = timezone.now()
    alert.save()
    
    return Response({
        'message': 'Alert resolved successfully',
        'alert_id': str(alert.id),
        'resolved_at': alert.resolved_at.isoformat()
    })


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def test_alert_rule(request, rule_id):
    """Test an alert rule with sample data."""
    from .services import alert_generator
    
    rule = get_object_or_404(AlertRule, id=rule_id, owner=request.user)
    
    # Get sample data from request or use defaults
    sample_data = request.data.get('sample_data', {})
    
    # Create sample transaction data
    sample_transaction_data = {
        'transaction_id': sample_data.get('transaction_id', 'test_txn_123'),
        'user_id': sample_data.get('user_id', 'test_user_456'),
        'amount': float(sample_data.get('amount', 1000.0)),
        'currency': sample_data.get('currency', 'USD'),
        'merchant_id': sample_data.get('merchant_id', 'test_merchant'),
        'payment_method': sample_data.get('payment_method', 'credit_card'),
        'ip_address': sample_data.get('ip_address', '192.168.1.1'),
        'timestamp': timezone.now()
    }
    
    # Create sample fraud result
    sample_fraud_result = {
        'fraud_probability': float(sample_data.get('fraud_score', 0.75)),
        'risk_score': float(sample_data.get('risk_score', 0.8)),
        'risk_level': sample_data.get('risk_level', 'high'),
        'confidence': float(sample_data.get('confidence', 0.9)),
        'ml_results': {},
        'rule_results': []
    }
    
    # Create alert context for testing
    alert_context = alert_generator._create_alert_context(
        type('Transaction', (), sample_transaction_data)(),
        sample_fraud_result
    )
    
    # Test rule evaluation
    rule_triggered = alert_generator._evaluate_rule(rule, alert_context)
    
    return Response({
        'rule_id': str(rule.id),
        'rule_name': rule.name,
        'rule_triggered': rule_triggered,
        'sample_data_used': sample_transaction_data,
        'fraud_result_used': sample_fraud_result,
        'alert_context': {
            'fraud_score': alert_context.get('fraud_score'),
            'amount': alert_context.get('amount'),
            'derived_features': {
                'is_high_amount': alert_context.get('is_high_amount'),
                'fraud_score_category': alert_context.get('fraud_score_category'),
                'amount_category': alert_context.get('amount_category')
            }
        }
    })


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def test_notification_channel(request):
    """Test a notification channel."""
    # This is a placeholder - will be implemented in later tasks
    return Response({
        'message': 'Channel test functionality will be implemented in tasks 5-8'
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def alert_dashboard(request):
    """Get alert dashboard data."""
    user_alerts = Alert.objects.filter(owner=request.user)
    
    dashboard_data = {
        'total_alerts': user_alerts.count(),
        'pending_alerts': user_alerts.filter(status='pending').count(),
        'acknowledged_alerts': user_alerts.filter(status='acknowledged').count(),
        'resolved_alerts': user_alerts.filter(status='resolved').count(),
        'critical_alerts': user_alerts.filter(severity='critical').count(),
        'high_alerts': user_alerts.filter(severity='high').count(),
        'medium_alerts': user_alerts.filter(severity='medium').count(),
        'low_alerts': user_alerts.filter(severity='low').count(),
    }
    
    return Response(dashboard_data)