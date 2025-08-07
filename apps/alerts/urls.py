from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'alerts', views.AlertViewSet)
router.register(r'rules', views.AlertRuleViewSet)
router.register(r'deliveries', views.NotificationDeliveryViewSet)
router.register(r'escalation-rules', views.EscalationRuleViewSet)
router.register(r'templates', views.NotificationTemplateViewSet)
router.register(r'escalation-tasks', views.EscalationTaskViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('alerts/<uuid:alert_id>/acknowledge/', views.acknowledge_alert, name='acknowledge_alert'),
    path('alerts/<uuid:alert_id>/resolve/', views.resolve_alert, name='resolve_alert'),
    path('rules/<uuid:rule_id>/test/', views.test_alert_rule, name='test_alert_rule'),
    path('channels/test/', views.test_notification_channel, name='test_notification_channel'),
    path('dashboard/', views.alert_dashboard, name='alert_dashboard'),
]