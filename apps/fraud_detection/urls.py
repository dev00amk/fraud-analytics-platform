from django.urls import path
from . import views

urlpatterns = [
    path('analyze/', views.FraudAnalysisView.as_view(), name='fraud_analysis'),
    path('rules/', views.FraudRuleListCreateView.as_view(), name='fraud_rules'),
    path('alerts/', views.FraudAlertListView.as_view(), name='fraud_alerts'),
]