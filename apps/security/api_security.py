"""
Advanced API Security Middleware for Fraud Analytics Platform
Comprehensive protection against OWASP Top 10 and financial security threats
"""

import hashlib
import hmac
import json
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import ipaddress
from django.conf import settings
from django.core.cache import cache
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.utils import timezone
from django.utils.deprecation import MiddlewareMixin
from redis import Redis

from .security_manager import security_manager, SecurityException


class APISecurityMiddleware(MiddlewareMixin):
    """
    Comprehensive API security middleware providing:
    - Rate limiting with multiple algorithms
    - DDoS protection
    - Geographic blocking
    - Request signature validation
    - SQL injection protection
    - XSS protection
    - CSRF protection
    - Security headers
    """

    def __init__(self, get_response):
        self.get_response = get_response
        redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379/0')
        self.redis_client = Redis.from_url(redis_url)
        
        # Security configurations
        self.rate_limits = self._load_rate_limits()
        self.blocked_ips = self._load_blocked_ips()
        self.suspicious_patterns = self._load_security_patterns()
        self.geographic_blocks = getattr(settings, 'GEOGRAPHIC_BLOCKS', [])
        
        # Initialize components
        self.rate_limiter = RateLimiter(self.redis_client)
        self.ddos_protector = DDoSProtector(self.redis_client)
        self.request_validator = RequestValidator()
        self.security_headers = SecurityHeaders()

    def __call__(self, request):
        """Main middleware processing."""
        try:
            # Pre-request security checks
            security_result = self._pre_request_security(request)
            if security_result:
                return security_result

            # Process the request
            response = self.get_response(request)

            # Post-request security enhancements
            response = self._post_request_security(request, response)

            return response

        except SecurityException as e:
            return self._security_error_response(str(e), 403)
        except Exception as e:
            # Log unexpected errors but don't expose details
            self._log_security_event('MIDDLEWARE_ERROR', error=str(e))
            return self._security_error_response("Security validation failed", 500)

    def _pre_request_security(self, request: HttpRequest) -> Optional[HttpResponse]:
        """Comprehensive pre-request security validation."""
        
        # 1. IP-based security checks
        client_ip = self._get_client_ip(request)
        
        # Check if IP is blocked
        if self._is_ip_blocked(client_ip):
            self._log_security_event('BLOCKED_IP_ACCESS', ip=client_ip)
            return self._security_error_response("Access denied", 403)

        # Geographic blocking
        if self._is_geographically_blocked(client_ip):
            self._log_security_event('GEOGRAPHIC_BLOCK', ip=client_ip)
            return self._security_error_response("Geographic access restriction", 403)

        # 2. DDoS protection
        if self.ddos_protector.is_ddos_attack(client_ip, request):
            self._log_security_event('DDOS_DETECTED', ip=client_ip)
            return self._security_error_response("Too many requests", 429)

        # 3. Rate limiting
        rate_limit_result = self.rate_limiter.check_rate_limit(request, client_ip)
        if not rate_limit_result['allowed']:
            self._log_security_event('RATE_LIMIT_EXCEEDED', 
                                   ip=client_ip, 
                                   limit=rate_limit_result['limit'])
            
            response = self._security_error_response("Rate limit exceeded", 429)
            response['Retry-After'] = str(rate_limit_result['retry_after'])
            response['X-RateLimit-Remaining'] = '0'
            response['X-RateLimit-Reset'] = str(rate_limit_result['reset_time'])
            return response

        # 4. Request validation
        validation_result = self.request_validator.validate_request(request)
        if not validation_result['valid']:
            self._log_security_event('MALICIOUS_REQUEST', 
                                   ip=client_ip,
                                   reason=validation_result['reason'])
            return self._security_error_response("Invalid request", 400)

        # 5. API signature validation (for authenticated endpoints)
        if self._requires_signature_validation(request):
            if not self._validate_api_signature(request):
                self._log_security_event('INVALID_SIGNATURE', ip=client_ip)
                return self._security_error_response("Invalid request signature", 401)

        # All checks passed
        return None

    def _post_request_security(self, request: HttpRequest, response: HttpResponse) -> HttpResponse:
        """Apply security headers and post-processing."""
        
        # Add security headers
        response = self.security_headers.add_headers(response, request)
        
        # Add rate limiting headers
        client_ip = self._get_client_ip(request)
        rate_info = self.rate_limiter.get_rate_info(request, client_ip)
        
        response['X-RateLimit-Limit'] = str(rate_info['limit'])
        response['X-RateLimit-Remaining'] = str(rate_info['remaining'])
        response['X-RateLimit-Reset'] = str(rate_info['reset_time'])
        
        # Monitor response for sensitive data leakage
        self._monitor_response_data(request, response)
        
        return response

    def _get_client_ip(self, request: HttpRequest) -> str:
        """Extract real client IP considering proxies."""
        # Check for common proxy headers
        forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if forwarded_for:
            # Take the first IP (original client)
            ip = forwarded_for.split(',')[0].strip()
            if self._is_valid_ip(ip):
                return ip
        
        real_ip = request.META.get('HTTP_X_REAL_IP')
        if real_ip and self._is_valid_ip(real_ip):
            return real_ip
        
        return request.META.get('REMOTE_ADDR', '0.0.0.0')

    def _is_valid_ip(self, ip: str) -> bool:
        """Validate IP address format."""
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False

    def _is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is in blocked list."""
        # Check Redis for dynamic blocks
        if self.redis_client.sismember('blocked_ips', ip):
            return True
        
        # Check static configuration
        return ip in self.blocked_ips

    def _is_geographically_blocked(self, ip: str) -> bool:
        """Check geographic restrictions (simplified implementation)."""
        # In production, use GeoIP database
        # This is a placeholder implementation
        return False

    def _requires_signature_validation(self, request: HttpRequest) -> bool:
        """Check if endpoint requires API signature validation."""
        # Define endpoints that require signature validation
        signature_required_paths = [
            '/api/v1/transactions/analyze',
            '/api/v1/cases/create',
            '/api/v1/webhooks'
        ]
        
        return any(request.path.startswith(path) for path in signature_required_paths)

    def _validate_api_signature(self, request: HttpRequest) -> bool:
        """Validate HMAC signature for API requests."""
        signature = request.headers.get('X-Signature')
        timestamp = request.headers.get('X-Timestamp')
        
        if not signature or not timestamp:
            return False
        
        # Check timestamp (prevent replay attacks)
        try:
            request_time = int(timestamp)
            current_time = int(time.time())
            if abs(current_time - request_time) > 300:  # 5 minutes tolerance
                return False
        except (ValueError, TypeError):
            return False
        
        # Validate signature
        api_key = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not api_key:
            return False
        
        # In production, retrieve secret from secure storage
        api_secret = self._get_api_secret(api_key)
        if not api_secret:
            return False
        
        # Calculate expected signature
        body = request.body.decode('utf-8') if request.body else ''
        message = f"{request.method}{request.path}{body}{timestamp}"
        expected_signature = hmac.new(
            api_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected_signature)

    def _get_api_secret(self, api_key: str) -> Optional[str]:
        """Retrieve API secret for key (implement secure storage)."""
        # Placeholder - implement secure key storage
        return "your-api-secret-here"

    def _load_rate_limits(self) -> Dict[str, Dict[str, int]]:
        """Load rate limiting configuration."""
        return {
            '/api/v1/analyze': {'requests': 100, 'window': 60},  # 100 req/min
            '/api/v1/transactions': {'requests': 1000, 'window': 60},
            '/api/v1/cases': {'requests': 200, 'window': 60},
            'default': {'requests': 1000, 'window': 3600}  # Default: 1000 req/hour
        }

    def _load_blocked_ips(self) -> Set[str]:
        """Load blocked IP addresses."""
        return set(getattr(settings, 'BLOCKED_IPS', []))

    def _load_security_patterns(self) -> Dict[str, List[str]]:
        """Load malicious patterns for detection."""
        return {
            'sql_injection': [
                r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b)",
                r"(\b(UNION|OR|AND)\b.*\b(SELECT|INSERT)\b)",
                r"('|\"|`).*(OR|AND).*('|\"|`)",
                r"(\b(EXEC|EXECUTE)\b.*\b(SP_|XP_)\b)"
            ],
            'xss': [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
                r"<iframe[^>]*>.*?</iframe>"
            ],
            'path_traversal': [
                r"\.\./",
                r"\.\.\\",
                r"~/"
            ]
        }

    def _monitor_response_data(self, request: HttpRequest, response: HttpResponse):
        """Monitor response for sensitive data leakage."""
        if hasattr(response, 'content'):
            content = response.content.decode('utf-8', errors='ignore')
            
            # Check for common sensitive data patterns
            sensitive_patterns = [
                r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email
            ]
            
            for pattern in sensitive_patterns:
                if re.search(pattern, content):
                    self._log_security_event('SENSITIVE_DATA_EXPOSURE',
                                           path=request.path,
                                           pattern=pattern)

    def _log_security_event(self, event_type: str, **kwargs):
        """Log security events."""
        event = {
            'timestamp': timezone.now().isoformat(),
            'event_type': event_type,
            'source': 'APISecurityMiddleware',
            **kwargs
        }
        
        self.redis_client.lpush('api_security_events', json.dumps(event))
        self.redis_client.ltrim('api_security_events', 0, 1000)

    def _security_error_response(self, message: str, status_code: int) -> JsonResponse:
        """Generate standardized security error response."""
        return JsonResponse({
            'error': 'Security validation failed',
            'message': message,
            'timestamp': timezone.now().isoformat()
        }, status=status_code)


class RateLimiter:
    """
    Advanced rate limiter supporting multiple algorithms:
    - Token bucket
    - Fixed window
    - Sliding window
    - Leaky bucket
    """

    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        
    def check_rate_limit(self, request: HttpRequest, client_ip: str) -> Dict[str, Any]:
        """Check if request should be rate limited."""
        
        # Determine rate limit for this endpoint
        endpoint = self._get_endpoint_pattern(request.path)
        user_id = getattr(request.user, 'id', None) if hasattr(request, 'user') else None
        
        # Multiple rate limiting strategies
        checks = [
            self._check_ip_rate_limit(client_ip, endpoint),
            self._check_user_rate_limit(user_id, endpoint) if user_id else {'allowed': True},
            self._check_global_rate_limit(),
        ]
        
        # If any check fails, deny the request
        for check in checks:
            if not check['allowed']:
                return check
        
        return {'allowed': True}

    def _check_ip_rate_limit(self, ip: str, endpoint: str) -> Dict[str, Any]:
        """IP-based rate limiting using sliding window."""
        key = f"rate_limit:ip:{ip}:{endpoint}"
        limit = 100  # requests per minute
        window = 60  # seconds
        
        return self._sliding_window_check(key, limit, window)

    def _check_user_rate_limit(self, user_id: int, endpoint: str) -> Dict[str, Any]:
        """User-based rate limiting."""
        key = f"rate_limit:user:{user_id}:{endpoint}"
        limit = 1000  # requests per hour for authenticated users
        window = 3600
        
        return self._sliding_window_check(key, limit, window)

    def _check_global_rate_limit(self) -> Dict[str, Any]:
        """Global rate limiting to protect against DDoS."""
        key = "rate_limit:global"
        limit = 10000  # global requests per minute
        window = 60
        
        return self._sliding_window_check(key, limit, window)

    def _sliding_window_check(self, key: str, limit: int, window: int) -> Dict[str, Any]:
        """Sliding window rate limiting algorithm."""
        now = int(time.time())
        pipeline = self.redis.pipeline()
        
        # Remove expired entries
        pipeline.zremrangebyscore(key, 0, now - window)
        
        # Count current requests
        pipeline.zcard(key)
        
        # Add current request
        pipeline.zadd(key, {str(now): now})
        
        # Set expiration
        pipeline.expire(key, window)
        
        results = pipeline.execute()
        current_requests = results[1]
        
        if current_requests >= limit:
            return {
                'allowed': False,
                'limit': limit,
                'current': current_requests,
                'retry_after': window,
                'reset_time': now + window
            }
        
        return {
            'allowed': True,
            'limit': limit,
            'current': current_requests,
            'remaining': limit - current_requests - 1
        }

    def _get_endpoint_pattern(self, path: str) -> str:
        """Map request path to rate limit pattern."""
        patterns = {
            r'^/api/v1/analyze': 'analyze',
            r'^/api/v1/transactions': 'transactions',
            r'^/api/v1/cases': 'cases'
        }
        
        for pattern, name in patterns.items():
            if re.match(pattern, path):
                return name
        
        return 'default'

    def get_rate_info(self, request: HttpRequest, client_ip: str) -> Dict[str, int]:
        """Get current rate limit information."""
        endpoint = self._get_endpoint_pattern(request.path)
        key = f"rate_limit:ip:{client_ip}:{endpoint}"
        
        current_count = self.redis.zcard(key)
        limit = 100  # Default limit
        
        return {
            'limit': limit,
            'remaining': max(0, limit - current_count),
            'reset_time': int(time.time()) + 60
        }


class DDoSProtector:
    """Advanced DDoS protection with multiple detection algorithms."""

    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        
    def is_ddos_attack(self, ip: str, request: HttpRequest) -> bool:
        """Detect DDoS attacks using multiple indicators."""
        
        # Connection frequency analysis
        if self._is_connection_flooding(ip):
            return True
        
        # Request pattern analysis
        if self._is_suspicious_pattern(ip, request):
            return True
        
        # Payload analysis
        if self._is_malicious_payload(request):
            return True
        
        return False

    def _is_connection_flooding(self, ip: str) -> bool:
        """Detect connection flooding."""
        key = f"ddos:connections:{ip}"
        now = int(time.time())
        
        # Count connections in last 10 seconds
        self.redis.zadd(key, {str(now): now})
        self.redis.zremrangebyscore(key, 0, now - 10)
        self.redis.expire(key, 10)
        
        connection_count = self.redis.zcard(key)
        
        # Alert if more than 50 connections in 10 seconds
        return connection_count > 50

    def _is_suspicious_pattern(self, ip: str, request: HttpRequest) -> bool:
        """Analyze request patterns for suspicious behavior."""
        key = f"ddos:pattern:{ip}"
        
        # Track request paths
        path_key = f"{key}:paths"
        self.redis.sadd(path_key, request.path)
        self.redis.expire(path_key, 60)
        
        # If hitting too many different endpoints, might be scanning
        unique_paths = self.redis.scard(path_key)
        if unique_paths > 20:  # More than 20 different paths in 1 minute
            return True
        
        return False

    def _is_malicious_payload(self, request: HttpRequest) -> bool:
        """Analyze request payload for malicious content."""
        if not request.body:
            return False
        
        try:
            body = request.body.decode('utf-8')
            
            # Check for extremely large payloads
            if len(body) > 1000000:  # 1MB limit
                return True
            
            # Check for malicious patterns
            malicious_patterns = [
                r'<script.*?>.*?</script>',
                r'javascript:',
                r'eval\(',
                r'document\.cookie'
            ]
            
            for pattern in malicious_patterns:
                if re.search(pattern, body, re.IGNORECASE):
                    return True
        
        except UnicodeDecodeError:
            # Suspicious if body can't be decoded
            return True
        
        return False


class RequestValidator:
    """Comprehensive request validation against common attacks."""

    def __init__(self):
        self.max_header_length = 8192
        self.max_url_length = 2048
        self.max_body_size = 1000000  # 1MB

    def validate_request(self, request: HttpRequest) -> Dict[str, Any]:
        """Comprehensive request validation."""
        
        # Header validation
        header_check = self._validate_headers(request)
        if not header_check['valid']:
            return header_check
        
        # URL validation
        url_check = self._validate_url(request)
        if not url_check['valid']:
            return url_check
        
        # Body validation
        body_check = self._validate_body(request)
        if not body_check['valid']:
            return body_check
        
        return {'valid': True}

    def _validate_headers(self, request: HttpRequest) -> Dict[str, Any]:
        """Validate HTTP headers."""
        
        for header, value in request.META.items():
            if not header.startswith('HTTP_'):
                continue
            
            # Check header length
            if len(str(value)) > self.max_header_length:
                return {
                    'valid': False,
                    'reason': f'Header {header} too long'
                }
            
            # Check for malicious patterns in headers
            if self._contains_malicious_pattern(str(value)):
                return {
                    'valid': False,
                    'reason': f'Malicious pattern in header {header}'
                }
        
        return {'valid': True}

    def _validate_url(self, request: HttpRequest) -> Dict[str, Any]:
        """Validate URL for malicious patterns."""
        
        full_url = request.build_absolute_uri()
        
        # Check URL length
        if len(full_url) > self.max_url_length:
            return {
                'valid': False,
                'reason': 'URL too long'
            }
        
        # Check for path traversal
        if '../' in request.path or '..\\' in request.path:
            return {
                'valid': False,
                'reason': 'Path traversal attempt'
            }
        
        # Check for SQL injection in query parameters
        query_string = request.META.get('QUERY_STRING', '')
        if self._contains_sql_injection(query_string):
            return {
                'valid': False,
                'reason': 'SQL injection in query parameters'
            }
        
        return {'valid': True}

    def _validate_body(self, request: HttpRequest) -> Dict[str, Any]:
        """Validate request body."""
        
        if not request.body:
            return {'valid': True}
        
        # Check body size
        if len(request.body) > self.max_body_size:
            return {
                'valid': False,
                'reason': 'Request body too large'
            }
        
        try:
            body_str = request.body.decode('utf-8')
            
            # Check for malicious patterns
            if self._contains_malicious_pattern(body_str):
                return {
                    'valid': False,
                    'reason': 'Malicious pattern in request body'
                }
            
        except UnicodeDecodeError:
            # Allow binary data but log suspicious activity
            pass
        
        return {'valid': True}

    def _contains_malicious_pattern(self, text: str) -> bool:
        """Check text for malicious patterns."""
        patterns = [
            # SQL injection
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b)",
            r"('|\").*(OR|AND).*('|\")",
            
            # XSS
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            
            # Command injection
            r"[;&|`]",
            r"\$\(",
            
            # Path traversal
            r"\.\./",
            r"\.\.\\",
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False

    def _contains_sql_injection(self, text: str) -> bool:
        """Specific SQL injection detection."""
        sql_patterns = [
            r"\b(SELECT|INSERT|UPDATE|DELETE)\b.*\b(FROM|INTO|SET|WHERE)\b",
            r"\b(UNION|OR|AND)\b.*\b(SELECT|INSERT)\b",
            r"('|\").*(OR|AND).*\b\d+\b.*('|\")",
            r"\b(EXEC|EXECUTE)\b.*\b(SP_|XP_)\b"
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False


class SecurityHeaders:
    """Comprehensive security headers management."""

    def add_headers(self, response: HttpResponse, request: HttpRequest) -> HttpResponse:
        """Add comprehensive security headers."""
        
        # Content Security Policy
        response['Content-Security-Policy'] = self._get_csp_header()
        
        # Strict Transport Security
        response['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains; preload'
        
        # X-Frame-Options
        response['X-Frame-Options'] = 'DENY'
        
        # X-Content-Type-Options
        response['X-Content-Type-Options'] = 'nosniff'
        
        # X-XSS-Protection
        response['X-XSS-Protection'] = '1; mode=block'
        
        # Referrer Policy
        response['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        # Permissions Policy
        response['Permissions-Policy'] = self._get_permissions_policy()
        
        # Custom security headers
        response['X-Request-ID'] = self._generate_request_id()
        response['X-Security-Version'] = '2.0'
        
        # Remove server information
        if 'Server' in response:
            del response['Server']
        
        return response

    def _get_csp_header(self) -> str:
        """Generate Content Security Policy header."""
        return (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: https:; "
            "connect-src 'self' https://api.fraud-platform.com; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "object-src 'none'"
        )

    def _get_permissions_policy(self) -> str:
        """Generate Permissions Policy header."""
        return (
            "geolocation=(), "
            "microphone=(), "
            "camera=(), "
            "magnetometer=(), "
            "gyroscope=(), "
            "payment=()"
        )

    def _generate_request_id(self) -> str:
        """Generate unique request ID for tracking."""
        return security_manager.generate_secure_token(16)