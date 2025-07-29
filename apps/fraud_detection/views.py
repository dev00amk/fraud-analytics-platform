from rest_framework import generics, status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import FraudAlert, FraudRule
from .serializers import FraudAlertSerializer, FraudRuleSerializer
from .services import FraudDetectionService


class FraudAnalysisView(generics.CreateAPIView):
    """Main fraud analysis endpoint."""

    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        fraud_service = FraudDetectionService()
        result = fraud_service.analyze_transaction(request.data, request.user)
        return Response(result, status=status.HTTP_200_OK)


class FraudRuleListCreateView(generics.ListCreateAPIView):
    """Fraud rules management."""

    serializer_class = FraudRuleSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return FraudRule.objects.filter(owner=self.request.user)


class FraudAlertListView(generics.ListAPIView):
    """Fraud alerts listing."""

    serializer_class = FraudAlertSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return FraudAlert.objects.filter(owner=self.request.user)
