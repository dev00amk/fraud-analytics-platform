from rest_framework import generics
from rest_framework.permissions import IsAuthenticated

from .models import Webhook
from .serializers import WebhookSerializer


class WebhookListCreateView(generics.ListCreateAPIView):
    """List and create webhooks."""

    serializer_class = WebhookSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Webhook.objects.filter(owner=self.request.user)
