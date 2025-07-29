"""
Production-grade security middleware for fraud analytics platform.
Implements rate limiting, idempotency, JWT blacklisting, and request validation.
"""

import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import redis
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.utils.deprecation import MiddlewareMixin
from rest_framework_simplejwt.exceptions import InvalidToken, TokenError
from rest_framework_simplejwt.tokens import UntypedToken

logger = logging.getLogger(__name__)
User = get_user_model()


class RateLimitMiddleware(MiddlewareMixin):
    """
    Advanced rate limiting middleware with per-IP and per-user limits.
    Implements sliding window rate limiting with Redis backend.
    """

    def __init__(self, get_response):
        super().__init__(get_response)
        self.redis_client = self._get_redis_client()

        # Rate limit configurations
        self.rate_limits = {
            "default": {"requests": 60, "window": 60},  # 60 req/min
            "auth": {"requests": 10, "window": 60},  # 10 req/min for auth endpoints
            "analyze": {"requests": 100, "window": 60},  # 100 req/min for analysis
            "webhook": {"requests": 1000, "window": 60},  # 1000 req/min for webhooks
        }

    def _get_redis_client(self):
        """Initialize Redis client for rate limiting."""
        try:
            return redis.Redis(
                host=getattr(settings, "REDIS_HOST", "localhost"),
                port=getattr(settings, "REDIS_PORT", 6379),
                db=getattr(settings, "REDIS_RATE_LIMIT_DB", 1),
                decode_responses=True,
            )
        except Exception as e:
            logger.warning(f"Redis connection failed for rate limiting: {e}")
            return None

    def process_request(self, request: HttpRequest) -> Optional[HttpResponse]:
        """Process incoming request for rate limiting."""
        if not self.redis_client:
            return None

        # Get client identifier
        client_id = self._get_client_id(request)

        # Determine rate limit category
        limit_category = self._get_limit_category(request.path)
        rate_config = self.rate_limits.get(limit_category, self.rate_limits["default"])

        # Check rate limit
        if self._is_rate_limited(client_id, limit_category, rate_config):
            return JsonResponse(
                {
                    "error": "Rate limit exceeded",
                    "detail": f'Maximum {rate_config["requests"]} requests per {rate_config["window"]} seconds',
                    "retry_after": rate_config["window"],
                },
                status=429,
            )

        return None

    def _get_client_id(self, request: HttpRequest) -> str:
        """Get unique client identifier for rate limiting."""
        # Try to get user ID first
        if hasattr(request, "user") and request.user.is_authenticated:
            return f"user:{request.user.id}"

        # Fall back to IP address
        x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
        if x_forwarded_for:
            ip = x_forwarded_for.split(",")[0].strip()
        else:
            ip = request.META.get("REMOTE_ADDR", "unknown")

        return f"ip:{ip}"

    def _get_limit_category(self, path: str) -> str:
        """Determine rate limit category based on request path."""
        if "/auth/" in path:
            return "auth"
        elif "/fraud/analyze" in path:
            return "analyze"
        elif "/webhooks/" in path:
            return "webhook"
        else:
            return "default"

    def _is_rate_limited(self, client_id: str, category: str, config: Dict) -> bool:
        """Check if client is rate limited using sliding window."""
        try:
            key = f"rate_limit:{category}:{client_id}"
            current_time = int(time.time())
            window_start = current_time - config["window"]

            # Remove old entries
            self.redis_client.zremrangebyscore(key, 0, window_start)

            # Count current requests
            current_requests = self.redis_client.zcard(key)

            if current_requests >= config["requests"]:
                return True

            # Add current request
            self.redis_client.zadd(key, {str(current_time): current_time})
            self.redis_client.expire(key, config["window"])

            return False

        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            return False  # Fail open


class IdempotencyMiddleware(MiddlewareMixin):
    """
    Idempotency middleware to handle duplicate transaction submissions.
    Uses Redis to track request fingerprints and prevent duplicate processing.
    """

    def __init__(self, get_response):
        super().__init__(get_response)
        self.redis_client = self._get_redis_client()
        self.idempotent_methods = {"POST", "PUT", "PATCH"}
        self.idempotent_paths = ["/api/v1/transactions/", "/api/v1/fraud/analyze/"]
        self.ttl = 3600  # 1 hour

    def _get_redis_client(self):
        """Initialize Redis client for idempotency."""
        try:
            return redis.Redis(
                host=getattr(settings, "REDIS_HOST", "localhost"),
                port=getattr(settings, "REDIS_PORT", 6379),
                db=getattr(settings, "REDIS_IDEMPOTENCY_DB", 2),
                decode_responses=False,  # Keep binary for JSON storage
            )
        except Exception as e:
            logger.warning(f"Redis connection failed for idempotency: {e}")
            return None

    def process_request(self, request: HttpRequest) -> Optional[HttpResponse]:
        """Process request for idempotency check."""
        if not self._should_check_idempotency(request):
            return None

        if not self.redis_client:
            return None

        # Generate request fingerprint
        fingerprint = self._generate_fingerprint(request)

        # Check for existing response
        cached_response = self._get_cached_response(fingerprint)
        if cached_response:
            logger.info(
                f"Returning cached response for duplicate request: {fingerprint}"
            )
            return JsonResponse(
                cached_response["data"],
                status=cached_response["status"],
                headers={"X-Idempotency-Replay": "true"},
            )

        # Store fingerprint for response caching
        request._idempotency_key = fingerprint
        return None

    def process_response(
        self, request: HttpRequest, response: HttpResponse
    ) -> HttpResponse:
        """Cache successful responses for idempotency."""
        if not hasattr(request, "_idempotency_key"):
            return response

        if not self.redis_client:
            return response

        # Only cache successful responses
        if 200 <= response.status_code < 300:
            try:
                response_data = {
                    "data": json.loads(response.content.decode("utf-8")),
                    "status": response.status_code,
                    "timestamp": datetime.now().isoformat(),
                }

                self.redis_client.setex(
                    f"idempotency:{request._idempotency_key}",
                    self.ttl,
                    json.dumps(response_data),
                )

            except Exception as e:
                logger.error(f"Failed to cache idempotent response: {e}")

        return response

    def _should_check_idempotency(self, request: HttpRequest) -> bool:
        """Determine if request should be checked for idempotency."""
        return request.method in self.idempotent_methods and any(
            path in request.path for path in self.idempotent_paths
        )

    def _generate_fingerprint(self, request: HttpRequest) -> str:
        """Generate unique fingerprint for request."""
        # Include user, path, method, and body content
        user_id = (
            str(request.user.id)
            if hasattr(request, "user") and request.user.is_authenticated
            else "anonymous"
        )

        fingerprint_data = {
            "user_id": user_id,
            "method": request.method,
            "path": request.path,
            "body": request.body.decode("utf-8") if request.body else "",
            "query_params": dict(request.GET),
        }

        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()

    def _get_cached_response(self, fingerprint: str) -> Optional[Dict]:
        """Get cached response for fingerprint."""
        try:
            cached = self.redis_client.get(f"idempotency:{fingerprint}")
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.error(f"Failed to retrieve cached response: {e}")

        return None


class JWTBlacklistMiddleware(MiddlewareMixin):
    """
    JWT blacklist middleware to handle token revocation.
    Maintains blacklisted tokens in Redis with automatic expiration.
    """

    def __init__(self, get_response):
        super().__init__(get_response)
        self.redis_client = self._get_redis_client()

    def _get_redis_client(self):
        """Initialize Redis client for JWT blacklist."""
        try:
            return redis.Redis(
                host=getattr(settings, "REDIS_HOST", "localhost"),
                port=getattr(settings, "REDIS_PORT", 6379),
                db=getattr(settings, "REDIS_JWT_BLACKLIST_DB", 3),
                decode_responses=True,
            )
        except Exception as e:
            logger.warning(f"Redis connection failed for JWT blacklist: {e}")
            return None

    def process_request(self, request: HttpRequest) -> Optional[HttpResponse]:
        """Check if JWT token is blacklisted."""
        if not self.redis_client:
            return None

        # Extract JWT token from Authorization header
        auth_header = request.META.get("HTTP_AUTHORIZATION", "")
        if not auth_header.startswith("Bearer "):
            return None

        token = auth_header.split(" ")[1]

        # Check if token is blacklisted
        if self._is_token_blacklisted(token):
            return JsonResponse(
                {
                    "error": "Token has been revoked",
                    "detail": "Please obtain a new access token",
                },
                status=401,
            )

        return None

    def _is_token_blacklisted(self, token: str) -> bool:
        """Check if token is in blacklist."""
        try:
            # Validate token structure first
            UntypedToken(token)

            # Check blacklist
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            return self.redis_client.exists(f"blacklist:{token_hash}")

        except (InvalidToken, TokenError):
            # Invalid token structure
            return True
        except Exception as e:
            logger.error(f"JWT blacklist check error: {e}")
            return False  # Fail open

    def blacklist_token(self, token: str, expiry_seconds: int = 86400):
        """Add token to blacklist."""
        if not self.redis_client:
            return False

        try:
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            self.redis_client.setex(
                f"blacklist:{token_hash}", expiry_seconds, datetime.now().isoformat()
            )
            return True
        except Exception as e:
            logger.error(f"Failed to blacklist token: {e}")
            return False


class WebhookSecurityMiddleware(MiddlewareMixin):
    """
    Webhook security middleware with HMAC signature verification.
    Prevents replay attacks and validates webhook authenticity.
    """

    def __init__(self, get_response):
        super().__init__(get_response)
        self.webhook_paths = ["/api/v1/webhooks/"]
        self.max_timestamp_skew = 300  # 5 minutes

    def process_request(self, request: HttpRequest) -> Optional[HttpResponse]:
        """Validate webhook security."""
        if not any(path in request.path for path in self.webhook_paths):
            return None

        if request.method != "POST":
            return None

        # Verify HMAC signature
        if not self._verify_hmac_signature(request):
            return JsonResponse(
                {
                    "error": "Invalid webhook signature",
                    "detail": "HMAC signature verification failed",
                },
                status=401,
            )

        # Check timestamp to prevent replay attacks
        if not self._verify_timestamp(request):
            return JsonResponse(
                {
                    "error": "Request timestamp invalid",
                    "detail": f"Timestamp must be within {self.max_timestamp_skew} seconds",
                },
                status=401,
            )

        return None

    def _verify_hmac_signature(self, request: HttpRequest) -> bool:
        """Verify HMAC signature in webhook request."""
        try:
            signature_header = request.META.get("HTTP_X_WEBHOOK_SIGNATURE", "")
            timestamp_header = request.META.get("HTTP_X_WEBHOOK_TIMESTAMP", "")

            if not signature_header or not timestamp_header:
                return False

            # Get webhook secret (in production, this would be per-webhook)
            webhook_secret = getattr(settings, "WEBHOOK_SECRET", "default-secret")

            # Create expected signature
            payload = f"{timestamp_header}.{request.body.decode('utf-8')}"
            expected_signature = hmac.new(
                webhook_secret.encode(), payload.encode(), hashlib.sha256
            ).hexdigest()

            # Compare signatures
            return hmac.compare_digest(
                signature_header.replace("sha256=", ""), expected_signature
            )

        except Exception as e:
            logger.error(f"HMAC verification error: {e}")
            return False

    def _verify_timestamp(self, request: HttpRequest) -> bool:
        """Verify request timestamp to prevent replay attacks."""
        try:
            timestamp_header = request.META.get("HTTP_X_WEBHOOK_TIMESTAMP", "")
            if not timestamp_header:
                return False

            request_time = int(timestamp_header)
            current_time = int(time.time())

            # Check if timestamp is within acceptable range
            time_diff = abs(current_time - request_time)
            return time_diff <= self.max_timestamp_skew

        except (ValueError, TypeError) as e:
            logger.error(f"Timestamp verification error: {e}")
            return False


class RequestValidationMiddleware(MiddlewareMixin):
    """
    Request validation middleware for input sanitization and validation.
    Implements comprehensive security checks for all API endpoints.
    """

    def __init__(self, get_response):
        super().__init__(get_response)
        self.max_request_size = 10 * 1024 * 1024  # 10MB
        self.dangerous_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"vbscript:",
            r"onload\s*=",
            r"onerror\s*=",
            r"eval\s*\(",
            r"expression\s*\(",
        ]

    def process_request(self, request: HttpRequest) -> Optional[HttpResponse]:
        """Validate incoming request."""
        # Check request size
        if self._is_request_too_large(request):
            return JsonResponse(
                {
                    "error": "Request too large",
                    "detail": f"Maximum request size is {self.max_request_size} bytes",
                },
                status=413,
            )

        # Validate content type for POST/PUT/PATCH
        if request.method in ["POST", "PUT", "PATCH"]:
            if not self._is_valid_content_type(request):
                return JsonResponse(
                    {
                        "error": "Invalid content type",
                        "detail": "Content-Type must be application/json",
                    },
                    status=400,
                )

        # Check for malicious patterns
        if self._contains_malicious_content(request):
            logger.warning(
                f"Malicious content detected in request from {request.META.get('REMOTE_ADDR')}"
            )
            return JsonResponse(
                {
                    "error": "Invalid request content",
                    "detail": "Request contains potentially malicious content",
                },
                status=400,
            )

        return None

    def _is_request_too_large(self, request: HttpRequest) -> bool:
        """Check if request exceeds size limit."""
        try:
            content_length = int(request.META.get("CONTENT_LENGTH", 0))
            return content_length > self.max_request_size
        except (ValueError, TypeError):
            return False

    def _is_valid_content_type(self, request: HttpRequest) -> bool:
        """Validate content type for API requests."""
        content_type = request.META.get("CONTENT_TYPE", "")
        return content_type.startswith("application/json")

    def _contains_malicious_content(self, request: HttpRequest) -> bool:
        """Check for malicious patterns in request."""
        import re

        # Check request body
        if request.body:
            body_str = request.body.decode("utf-8", errors="ignore")
            for pattern in self.dangerous_patterns:
                if re.search(pattern, body_str, re.IGNORECASE):
                    return True

        # Check query parameters
        for key, value in request.GET.items():
            for pattern in self.dangerous_patterns:
                if re.search(pattern, f"{key}={value}", re.IGNORECASE):
                    return True

        return False


class SecurityHeadersMiddleware(MiddlewareMixin):
    """
    Security headers middleware to add protective HTTP headers.
    Implements OWASP recommended security headers.
    """

    def process_response(
        self, request: HttpRequest, response: HttpResponse
    ) -> HttpResponse:
        """Add security headers to response."""
        # Content Security Policy
        response["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' https:; "
            "connect-src 'self' ws: wss:; "
            "frame-ancestors 'none';"
        )

        # Other security headers
        response["X-Content-Type-Options"] = "nosniff"
        response["X-Frame-Options"] = "DENY"
        response["X-XSS-Protection"] = "1; mode=block"
        response["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

        # HSTS (only in production with HTTPS)
        if not settings.DEBUG and request.is_secure():
            response["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )

        return response
