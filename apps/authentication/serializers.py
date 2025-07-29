from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import APIKey

User = get_user_model()


class UserSerializer(serializers.ModelSerializer):
    """User serializer."""
    password = serializers.CharField(write_only=True)
    
    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'company_name', 'password', 'is_verified')
        extra_kwargs = {'password': {'write_only': True}}
    
    def create(self, validated_data):
        password = validated_data.pop('password')
        user = User.objects.create_user(**validated_data)
        user.set_password(password)
        user.save()
        return user


class APIKeySerializer(serializers.ModelSerializer):
    """API key serializer."""
    
    class Meta:
        model = APIKey
        fields = ('id', 'name', 'key', 'is_active', 'created_at', 'last_used')
        read_only_fields = ('key', 'created_at', 'last_used')
    
    def create(self, validated_data):
        validated_data['user'] = self.context['request'].user
        return super().create(validated_data)