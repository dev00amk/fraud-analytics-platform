from django.urls import path
from . import views

urlpatterns = [
    path('', views.WebhookListCreateView.as_view(), name='webhook_list'),
]