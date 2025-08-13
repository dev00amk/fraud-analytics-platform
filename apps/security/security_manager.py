"""
Enterprise Security Manager for Fraud Analytics Platform
Comprehensive authentication, authorization, and security controls
"""

import hashlib
import hmac
import json
import secrets
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import bcrypt
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.utils import timezone
from redis import Redis

User = get_user_model()


class SecurityManager:
    """
    Enterprise-grade security manager handling authentication,
    authorization, encryption, and security monitoring.
    """

    def __init__(self):
        redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379/0')
        self.redis_client = Redis.from_url(redis_url)
        self.encryption_key = self._get_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # Security thresholds
        self.MAX_LOGIN_ATTEMPTS = 5
        self.LOCKOUT_DURATION = 900  # 15 minutes
        self.SESSION_TIMEOUT = 3600  # 1 hour
        self.SUSPICIOUS_THRESHOLD = 3
        
    def _get_encryption_key(self) -> bytes:
        """Generate or retrieve encryption key for sensitive data."""
        key = getattr(settings, 'ENCRYPTION_KEY', None)
        if not key:
            # Generate new key if not exists
            key = Fernet.generate_key()
            # In production, store this securely (vault, env, etc.)
        return key if isinstance(key, bytes) else key.encode()

    def hash_password(self, password: str) -> str:
        """Securely hash password using bcrypt with salt."""
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure random token."""
        return secrets.token_urlsafe(length)

    def encrypt_sensitive_data(self, data: Union[str, Dict]) -> str:
        """Encrypt sensitive data using Fernet symmetric encryption."""
        if isinstance(data, dict):
            data = json.dumps(data)
        encrypted = self.cipher_suite.encrypt(data.encode())
        return encrypted.decode()

    def decrypt_sensitive_data(self, encrypted_data: str) -> Union[str, Dict]:
        """Decrypt sensitive data."""
        try:
            decrypted = self.cipher_suite.decrypt(encrypted_data.encode())
            decoded = decrypted.decode()
            try:
                return json.loads(decoded)
            except json.JSONDecodeError:
                return decoded
        except Exception as e:
            raise SecurityException(f"Decryption failed: {str(e)}")


class AuthenticationManager:
    """
    Advanced authentication manager with multi-factor support,
    rate limiting, and security monitoring.
    """

    def __init__(self, security_manager: SecurityManager):
        self.security = security_manager
        self.redis_client = security_manager.redis_client

    def authenticate_user(
        self, 
        username: str, 
        password: str, 
        ip_address: str,
        user_agent: str,
        mfa_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive user authentication with security monitoring.
        
        Returns:
            Authentication result with tokens and security info
        """
        auth_attempt_key = f"auth_attempts:{username}:{ip_address}"
        
        try:
            # Check for account lockout
            if self._is_account_locked(username, ip_address):
                self._log_security_event(
                    "ACCOUNT_LOCKED_ATTEMPT",
                    username=username,
                    ip_address=ip_address,
                    user_agent=user_agent
                )
                raise AuthenticationException("Account temporarily locked due to suspicious activity")

            # Increment attempt counter
            self._increment_auth_attempts(username, ip_address)

            # Validate user credentials
            user = self._validate_credentials(username, password)
            
            # Check if MFA is required
            if user.mfa_enabled and not mfa_token:
                return {
                    "status": "mfa_required",
                    "message": "Multi-factor authentication required",
                    "mfa_methods": user.get_mfa_methods()
                }

            # Verify MFA if provided
            if user.mfa_enabled and mfa_token:
                if not self._verify_mfa_token(user, mfa_token):
                    raise AuthenticationException("Invalid MFA token")

            # Generate session tokens
            tokens = self._generate_session_tokens(user)
            
            # Create secure session
            session_id = self._create_secure_session(user, ip_address, user_agent)
            
            # Reset failed attempts on successful auth
            self.redis_client.delete(auth_attempt_key)
            
            # Log successful authentication
            self._log_security_event(
                "SUCCESSFUL_AUTHENTICATION",
                username=username,
                ip_address=ip_address,
                user_agent=user_agent,
                session_id=session_id
            )

            return {
                "status": "success",
                "tokens": tokens,
                "session_id": session_id,
                "user_id": user.id,
                "permissions": self._get_user_permissions(user),
                "expires_at": (timezone.now() + timedelta(seconds=self.security.SESSION_TIMEOUT)).isoformat()
            }

        except AuthenticationException as e:
            self._log_security_event(
                "AUTHENTICATION_FAILED",
                username=username,
                ip_address=ip_address,
                user_agent=user_agent,
                error=str(e)
            )
            raise
        except Exception as e:
            self._log_security_event(
                "AUTHENTICATION_ERROR",
                username=username,
                ip_address=ip_address,
                error=str(e)
            )
            raise AuthenticationException("Authentication service temporarily unavailable")

    def _is_account_locked(self, username: str, ip_address: str) -> bool:
        """Check if account is locked due to failed attempts."""
        attempt_key = f"auth_attempts:{username}:{ip_address}"
        attempts = self.redis_client.get(attempt_key)
        
        if attempts and int(attempts) >= self.security.MAX_LOGIN_ATTEMPTS:
            # Check if lockout period has expired
            lockout_key = f"lockout:{username}:{ip_address}"
            lockout_time = self.redis_client.get(lockout_key)
            
            if lockout_time:
                return True
            else:
                # Set lockout period
                self.redis_client.setex(
                    lockout_key,
                    self.security.LOCKOUT_DURATION,
                    int(time.time())
                )
                return True
        
        return False

    def _increment_auth_attempts(self, username: str, ip_address: str):
        """Increment failed authentication attempts."""
        attempt_key = f"auth_attempts:{username}:{ip_address}"
        pipe = self.redis_client.pipeline()
        pipe.incr(attempt_key)
        pipe.expire(attempt_key, self.security.LOCKOUT_DURATION)
        pipe.execute()

    def _validate_credentials(self, username: str, password: str) -> User:
        """Validate user credentials securely."""
        try:
            user = User.objects.get(username=username, is_active=True)
        except User.DoesNotExist:
            # Use constant-time comparison to prevent timing attacks
            dummy_hash = "$2b$12$dummy.hash.to.prevent.timing.attacks.here"
            bcrypt.checkpw(b"dummy", dummy_hash.encode())
            raise AuthenticationException("Invalid credentials")

        if not self.security.verify_password(password, user.password):
            raise AuthenticationException("Invalid credentials")

        return user

    def _verify_mfa_token(self, user: User, token: str) -> bool:
        """Verify multi-factor authentication token."""
        # Implementation for TOTP, SMS, or backup codes
        # This is a simplified version - implement based on your MFA method
        return True  # Placeholder

    def _generate_session_tokens(self, user: User) -> Dict[str, str]:
        """Generate secure JWT tokens for session."""
        now = timezone.now()
        
        # Access token (short-lived)
        access_payload = {
            'user_id': user.id,
            'username': user.username,
            'exp': now + timedelta(minutes=15),
            'iat': now,
            'type': 'access',
            'jti': self.security.generate_secure_token(16)
        }
        
        # Refresh token (longer-lived)
        refresh_payload = {
            'user_id': user.id,
            'exp': now + timedelta(days=7),
            'iat': now,
            'type': 'refresh',
            'jti': self.security.generate_secure_token(16)
        }
        
        access_token = jwt.encode(access_payload, settings.SECRET_KEY, algorithm='HS256')
        refresh_token = jwt.encode(refresh_payload, settings.SECRET_KEY, algorithm='HS256')
        
        # Store refresh token securely
        self.redis_client.setex(
            f"refresh_token:{user.id}:{refresh_payload['jti']}",
            timedelta(days=7).total_seconds(),
            refresh_token
        )
        
        return {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'token_type': 'Bearer'
        }

    def _create_secure_session(self, user: User, ip_address: str, user_agent: str) -> str:
        """Create secure user session with metadata."""
        session_id = self.security.generate_secure_token(32)
        
        session_data = {
            'user_id': user.id,
            'username': user.username,
            'ip_address': ip_address,
            'user_agent': user_agent,
            'created_at': timezone.now().isoformat(),
            'last_activity': timezone.now().isoformat()
        }
        
        # Store encrypted session data
        encrypted_session = self.security.encrypt_sensitive_data(session_data)
        self.redis_client.setex(
            f"session:{session_id}",
            self.security.SESSION_TIMEOUT,
            encrypted_session
        )
        
        return session_id

    def _get_user_permissions(self, user: User) -> List[str]:
        """Get user permissions and roles."""
        permissions = []
        
        # Get user groups and permissions
        for group in user.groups.all():
            permissions.extend(group.permissions.values_list('codename', flat=True))
        
        # Add user-specific permissions
        permissions.extend(user.user_permissions.values_list('codename', flat=True))
        
        return list(set(permissions))

    def _log_security_event(self, event_type: str, **kwargs):
        """Log security events for monitoring and audit."""
        event = {
            'timestamp': timezone.now().isoformat(),
            'event_type': event_type,
            'source': 'AuthenticationManager',
            **kwargs
        }
        
        # Store in Redis for real-time monitoring
        self.redis_client.lpush('security_events', json.dumps(event))
        self.redis_client.ltrim('security_events', 0, 1000)  # Keep last 1000 events
        
        # Also log to Django logging system
        import logging
        security_logger = logging.getLogger('security')
        security_logger.info(f"Security Event: {event_type}", extra=event)


class AuthorizationManager:
    """
    Role-based access control (RBAC) and permission management.
    """

    def __init__(self, security_manager: SecurityManager):
        self.security = security_manager
        self.redis_client = security_manager.redis_client

    def check_permission(
        self, 
        user: User, 
        permission: str, 
        resource: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> bool:
        """
        Check if user has specific permission for resource.
        
        Args:
            user: User object
            permission: Permission code (e.g., 'transactions.view')
            resource: Optional resource identifier
            context: Additional context for permission check
        """
        # Check cache first
        cache_key = f"permission:{user.id}:{permission}:{resource or 'global'}"
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            return cached_result == 'allowed'

        # Perform permission check
        has_permission = self._evaluate_permission(user, permission, resource, context)
        
        # Cache result for 5 minutes
        cache.set(cache_key, 'allowed' if has_permission else 'denied', 300)
        
        # Log permission check
        self._log_authorization_event(
            'PERMISSION_CHECK',
            user_id=user.id,
            permission=permission,
            resource=resource,
            result='granted' if has_permission else 'denied'
        )
        
        return has_permission

    def _evaluate_permission(
        self, 
        user: User, 
        permission: str, 
        resource: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> bool:
        """Evaluate permission based on roles and rules."""
        # Super admin bypass
        if user.is_superuser:
            return True

        # Check direct permissions
        if user.user_permissions.filter(codename=permission).exists():
            return True

        # Check group permissions
        if user.groups.filter(permissions__codename=permission).exists():
            return True

        # Check role-based permissions
        for role in user.roles.all():
            if self._check_role_permission(role, permission, resource, context):
                return True

        return False

    def _check_role_permission(
        self, 
        role, 
        permission: str, 
        resource: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> bool:
        """Check role-specific permissions with context."""
        # Implement role-based logic here
        # This is a simplified version
        return role.permissions.filter(codename=permission).exists()

    def get_user_roles(self, user: User) -> List[str]:
        """Get all user roles."""
        roles = []
        
        # Add group-based roles
        roles.extend(user.groups.values_list('name', flat=True))
        
        # Add custom roles if you have a Role model
        if hasattr(user, 'roles'):
            roles.extend(user.roles.values_list('name', flat=True))
        
        return roles

    def _log_authorization_event(self, event_type: str, **kwargs):
        """Log authorization events."""
        event = {
            'timestamp': timezone.now().isoformat(),
            'event_type': event_type,
            'source': 'AuthorizationManager',
            **kwargs
        }
        
        self.redis_client.lpush('authorization_events', json.dumps(event))
        self.redis_client.ltrim('authorization_events', 0, 1000)


class SecurityMonitor:
    """
    Real-time security monitoring and threat detection.
    """

    def __init__(self, security_manager: SecurityManager):
        self.security = security_manager
        self.redis_client = security_manager.redis_client

    def monitor_user_behavior(
        self, 
        user_id: int, 
        action: str, 
        ip_address: str,
        user_agent: str,
        metadata: Optional[Dict] = None
    ):
        """Monitor and analyze user behavior for anomalies."""
        behavior_key = f"user_behavior:{user_id}"
        
        behavior_event = {
            'timestamp': timezone.now().isoformat(),
            'action': action,
            'ip_address': ip_address,
            'user_agent': user_agent,
            'metadata': metadata or {}
        }
        
        # Store behavior event
        self.redis_client.lpush(behavior_key, json.dumps(behavior_event))
        self.redis_client.ltrim(behavior_key, 0, 100)  # Keep last 100 events
        self.redis_client.expire(behavior_key, 86400)  # Expire after 24 hours
        
        # Analyze for suspicious patterns
        self._analyze_suspicious_behavior(user_id, behavior_event)

    def _analyze_suspicious_behavior(self, user_id: int, event: Dict):
        """Analyze behavior patterns for security threats."""
        behavior_key = f"user_behavior:{user_id}"
        recent_events = self.redis_client.lrange(behavior_key, 0, 9)  # Last 10 events
        
        events = [json.loads(e) for e in recent_events]
        
        # Check for rapid successive actions (potential bot)
        if len(events) >= 5:
            timestamps = [datetime.fromisoformat(e['timestamp']) for e in events[:5]]
            time_span = (timestamps[0] - timestamps[-1]).total_seconds()
            
            if time_span < 10:  # 5 actions in less than 10 seconds
                self._flag_suspicious_activity(
                    user_id, 
                    'RAPID_ACTIONS', 
                    f"5 actions in {time_span:.2f} seconds"
                )

        # Check for multiple IP addresses
        ip_addresses = set(e['ip_address'] for e in events)
        if len(ip_addresses) > 3:  # More than 3 different IPs in recent activity
            self._flag_suspicious_activity(
                user_id,
                'MULTIPLE_IPS',
                f"Activity from {len(ip_addresses)} different IP addresses"
            )

    def _flag_suspicious_activity(self, user_id: int, threat_type: str, details: str):
        """Flag suspicious activity and take appropriate action."""
        alert = {
            'timestamp': timezone.now().isoformat(),
            'user_id': user_id,
            'threat_type': threat_type,
            'details': details,
            'severity': 'HIGH'
        }
        
        # Store security alert
        self.redis_client.lpush('security_alerts', json.dumps(alert))
        
        # Log security event
        import logging
        security_logger = logging.getLogger('security')
        security_logger.warning(f"Suspicious Activity Detected: {threat_type}", extra=alert)
        
        # Take automated response if needed
        self._automated_security_response(user_id, threat_type)

    def _automated_security_response(self, user_id: int, threat_type: str):
        """Automated security responses to threats."""
        if threat_type in ['RAPID_ACTIONS', 'MULTIPLE_IPS']:
            # Temporarily rate limit user
            rate_limit_key = f"rate_limit:{user_id}"
            self.redis_client.setex(rate_limit_key, 300, 1)  # 5 minutes
            
            # Require re-authentication for sensitive actions
            sensitive_key = f"require_reauth:{user_id}"
            self.redis_client.setex(sensitive_key, 1800, 1)  # 30 minutes


# Custom Exceptions
class SecurityException(Exception):
    """Base security exception."""
    pass

class AuthenticationException(SecurityException):
    """Authentication-related exception."""
    pass

class AuthorizationException(SecurityException):
    """Authorization-related exception."""
    pass


# Initialize security components
security_manager = SecurityManager()
auth_manager = AuthenticationManager(security_manager)
authz_manager = AuthorizationManager(security_manager)
security_monitor = SecurityMonitor(security_manager)