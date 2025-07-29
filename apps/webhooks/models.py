import uuid

from django.contrib.auth import get_user_model
from django.db import models

User = get_user_model()


class Webhook(models.Model):
    """Webhook endpoints for notifications."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    url = models.URLField()
    secret = models.CharField(max_length=255)
    events = models.JSONField(default=list)  # List of events to subscribe to
    is_active = models.BooleanField(default=True)

    # Ownership
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name="webhooks")

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} - {self.url}"
