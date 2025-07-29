from rest_framework import generics, status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from apps.fraud_detection.services import FraudDetectionService

from .models import Transaction
from .serializers import TransactionAnalysisSerializer, TransactionSerializer


class TransactionListCreateView(generics.ListCreateAPIView):
    """List and create transactions."""

    serializer_class = TransactionSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Transaction.objects.filter(owner=self.request.user)


class TransactionDetailView(generics.RetrieveUpdateDestroyAPIView):
    """Retrieve, update, and delete transactions."""

    serializer_class = TransactionSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Transaction.objects.filter(owner=self.request.user)


class TransactionAnalysisView(generics.CreateAPIView):
    """Analyze transaction for fraud."""

    serializer_class = TransactionAnalysisSerializer
    permission_classes = [IsAuthenticated]

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        # Run fraud analysis
        fraud_service = FraudDetectionService()
        analysis_result = fraud_service.analyze_transaction(
            serializer.validated_data, request.user
        )

        return Response(analysis_result, status=status.HTTP_200_OK)
