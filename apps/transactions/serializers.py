from rest_framework import serializers
from .models import Transaction


class TransactionSerializer(serializers.ModelSerializer):
    """Transaction serializer."""
    
    class Meta:
        model = Transaction
        fields = [
            'id', 'transaction_id', 'user_id', 'amount', 'currency',
            'merchant_id', 'payment_method', 'status', 'fraud_score',
            'risk_level', 'ip_address', 'user_agent', 'device_fingerprint',
            'timestamp', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'fraud_score', 'risk_level', 'created_at', 'updated_at']
    
    def create(self, validated_data):
        validated_data['owner'] = self.context['request'].user
        return super().create(validated_data)


class TransactionAnalysisSerializer(serializers.Serializer):
    """Serializer for transaction fraud analysis."""
    transaction_id = serializers.CharField(max_length=255)
    user_id = serializers.CharField(max_length=255)
    amount = serializers.DecimalField(max_digits=12, decimal_places=2)
    currency = serializers.CharField(max_length=3, default='USD')
    merchant_id = serializers.CharField(max_length=255)
    payment_method = serializers.CharField(max_length=50)
    ip_address = serializers.IPAddressField()
    user_agent = serializers.CharField(required=False, allow_blank=True)
    device_fingerprint = serializers.CharField(required=False, allow_blank=True)
    timestamp = serializers.DateTimeField()