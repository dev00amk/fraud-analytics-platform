from rest_framework import serializers

from .models import FraudAlert, FraudRule


class FraudRuleSerializer(serializers.ModelSerializer):
    """Fraud rule serializer."""

    class Meta:
        model = FraudRule
        fields = [
            "id",
            "name",
            "description",
            "conditions",
            "action",
            "is_active",
            "priority",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]

    def create(self, validated_data):
        validated_data["owner"] = self.context["request"].user
        return super().create(validated_data)


class FraudAlertSerializer(serializers.ModelSerializer):
    """Fraud alert serializer."""

    class Meta:
        model = FraudAlert
        fields = [
            "id",
            "transaction_id",
            "alert_type",
            "severity",
            "message",
            "is_resolved",
            "created_at",
            "resolved_at",
        ]
        read_only_fields = ["id", "created_at"]
