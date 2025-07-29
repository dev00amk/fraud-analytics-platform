from django.urls import path
from . import views

urlpatterns = [
    path('dashboard/', views.DashboardStatsView.as_view(), name='dashboard_stats'),
]