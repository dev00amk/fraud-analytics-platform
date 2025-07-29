import uuid

from django.contrib.auth import get_user_model
from django.db import models

User = get_user_model()


class Transaction(models.Model):
    """Transaction model for fraud analysis."""

    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("approved", "Approved"),
        ("declined", "Declined"),
        ("flagged", "Flagged"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    transaction_id = models.CharField(max_length=255, unique=True)
    user_id = models.CharField(max_length=255)
    amount = models.DecimalField(max_digits=12, decimal_places=2)
    currency = models.CharField(max_length=3, default="USD")
    merchant_id = models.CharField(max_length=255)
    payment_method = models.CharField(max_length=50)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")

    # Fraud analysis fields
    fraud_score = models.FloatField(null=True, blank=True)
    risk_level = models.CharField(max_length=20, blank=True)

    # Request metadata
    ip_address = models.GenericIPAddressField()
    user_agent = models.TextField(blank=True)
    device_fingerprint = models.CharField(max_length=255, blank=True)

    # Timestamps
    timestamp = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Owner
    owner = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="transactions"
    )

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["transaction_id"]),
            models.Index(fields=["user_id"]),
            models.Index(fields=["status"]),
            models.Index(fields=["created_at"]),
        ]

    def __str__(self):
        return f"Transaction {self.transaction_id} - {self.amount} {self.currency}"
