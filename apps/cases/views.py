from rest_framework import generics
from rest_framework.permissions import IsAuthenticated
from .models import Case
from .serializers import CaseSerializer


class CaseListCreateView(generics.ListCreateAPIView):
    """List and create fraud cases."""
    serializer_class = CaseSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return Case.objects.filter(owner=self.request.user)


class CaseDetailView(generics.RetrieveUpdateDestroyAPIView):
    """Retrieve, update, and delete fraud cases."""
    serializer_class = CaseSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return Case.objects.filter(owner=self.request.user)