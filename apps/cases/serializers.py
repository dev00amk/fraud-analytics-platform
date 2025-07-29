from rest_framework import serializers

from .models import Case


class CaseSerializer(serializers.ModelSerializer):
    """Case serializer."""

    class Meta:
        model = Case
        fields = [
            "id",
            "case_number",
            "title",
            "description",
            "status",
            "priority",
            "transaction_id",
            "assigned_to",
            "created_at",
            "updated_at",
            "resolved_at",
        ]
        read_only_fields = ["id", "case_number", "created_at", "updated_at"]

    def create(self, validated_data):
        validated_data["owner"] = self.context["request"].user
        # Generate case number
        import uuid

        validated_data["case_number"] = f"CASE-{str(uuid.uuid4())[:8].upper()}"
        return super().create(validated_data)
