from django.urls import path
from . import views

urlpatterns = [
    path('', views.TransactionListCreateView.as_view(), name='transaction_list'),
    path('<uuid:pk>/', views.TransactionDetailView.as_view(), name='transaction_detail'),
    path('analyze/', views.TransactionAnalysisView.as_view(), name='transaction_analysis'),
]