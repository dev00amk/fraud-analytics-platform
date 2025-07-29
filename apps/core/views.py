import redis
from django.core.cache import cache
from django.db import connection
from django.http import JsonResponse


def health_check(request):
    """Health check endpoint for monitoring."""
    health_status = {"status": "healthy", "database": "unknown", "cache": "unknown"}

    # Check database
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        health_status["database"] = "healthy"
    except Exception:
        health_status["database"] = "unhealthy"
        health_status["status"] = "unhealthy"

    # Check cache/Redis
    try:
        cache.set("health_check", "ok", 30)
        if cache.get("health_check") == "ok":
            health_status["cache"] = "healthy"
        else:
            health_status["cache"] = "unhealthy"
            health_status["status"] = "unhealthy"
    except Exception:
        health_status["cache"] = "unhealthy"
        health_status["status"] = "unhealthy"

    status_code = 200 if health_status["status"] == "healthy" else 503
    return JsonResponse(health_status, status=status_code)
