from django.db.models import Avg
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.cases.models import Case
from apps.transactions.models import Transaction


class DashboardStatsView(APIView):
    """Dashboard statistics endpoint."""

    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user

        # Transaction stats
        total_transactions = Transaction.objects.filter(owner=user).count()
        flagged_transactions = Transaction.objects.filter(
            owner=user, status="flagged"
        ).count()

        # Case stats
        open_cases = Case.objects.filter(owner=user, status="open").count()

        # Fraud score average
        avg_fraud_score = (
            Transaction.objects.filter(owner=user, fraud_score__isnull=False).aggregate(
                avg_score=Avg("fraud_score")
            )["avg_score"]
            or 0
        )

        return Response(
            {
                "total_transactions": total_transactions,
                "flagged_transactions": flagged_transactions,
                "open_cases": open_cases,
                "average_fraud_score": round(avg_fraud_score, 2),
                "fraud_rate": round(
                    (flagged_transactions / max(total_transactions, 1)) * 100, 2
                ),
            }
        )
