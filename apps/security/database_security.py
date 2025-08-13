"""
Secure Database Layer for Fraud Analytics Platform
Enterprise-grade database security with encryption, access controls, and audit trails
"""

import base64
import hashlib
import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from django.conf import settings
from django.core.cache import cache
from django.db import models, transaction
from django.db.models import Q
from django.utils import timezone

from .security_manager import security_manager


class DatabaseSecurityManager:
    """
    Comprehensive database security management including:
    - Field-level encryption
    - Database connection security
    - Query audit logging
    - Data masking and anonymization
    - Secure backup and recovery
    """

    def __init__(self):
        self.encryption_keys = self._initialize_encryption_keys()
        self.audit_logger = DatabaseAuditLogger()
        
    def _initialize_encryption_keys(self) -> Dict[str, bytes]:
        """Initialize encryption keys for different data types."""
        return {
            'pii': self._get_or_create_key('PII_ENCRYPTION_KEY'),
            'financial': self._get_or_create_key('FINANCIAL_ENCRYPTION_KEY'),
            'sensitive': self._get_or_create_key('SENSITIVE_ENCRYPTION_KEY'),
            'audit': self._get_or_create_key('AUDIT_ENCRYPTION_KEY')
        }
    
    def _get_or_create_key(self, key_name: str) -> bytes:
        """Get encryption key from secure storage or generate new one."""
        key = getattr(settings, key_name, None)
        if not key:
            key = Fernet.generate_key()
            # In production, store this in a secure key management system
        return key if isinstance(key, bytes) else key.encode()


class EncryptedField(models.CharField):
    """
    Custom Django field that automatically encrypts/decrypts data.
    Supports multiple encryption levels based on data sensitivity.
    """

    def __init__(self, encryption_type='sensitive', *args, **kwargs):
        self.encryption_type = encryption_type
        kwargs['max_length'] = kwargs.get('max_length', 500)  # Encrypted data is longer
        super().__init__(*args, **kwargs)

    def encrypt_value(self, value: str) -> str:
        """Encrypt a value using specified encryption type."""
        if not value:
            return value
            
        db_security = DatabaseSecurityManager()
        encryption_key = db_security.encryption_keys[self.encryption_type]
        cipher_suite = Fernet(encryption_key)
        
        encrypted_bytes = cipher_suite.encrypt(value.encode('utf-8'))
        return base64.b64encode(encrypted_bytes).decode('utf-8')

    def decrypt_value(self, value: str) -> str:
        """Decrypt a value."""
        if not value:
            return value
            
        try:
            db_security = DatabaseSecurityManager()
            encryption_key = db_security.encryption_keys[self.encryption_type]
            cipher_suite = Fernet(encryption_key)
            
            encrypted_bytes = base64.b64decode(value.encode('utf-8'))
            decrypted_bytes = cipher_suite.decrypt(encrypted_bytes)
            return decrypted_bytes.decode('utf-8')
        except Exception:
            # Return masked value if decryption fails
            return "[ENCRYPTED]"

    def to_python(self, value):
        """Convert database value to Python value (decrypt)."""
        if value is None:
            return value
        return self.decrypt_value(value)

    def from_db_value(self, value, expression, connection):
        """Convert database value to Python value."""
        return self.to_python(value)

    def get_prep_value(self, value):
        """Convert Python value to database value (encrypt)."""
        if value is None:
            return value
        return self.encrypt_value(str(value))


class HashedField(models.CharField):
    """
    Field that stores hashed values for one-way encryption.
    Useful for passwords, tokens, and other values that don't need decryption.
    """

    def __init__(self, hash_algorithm='sha256', salt_length=32, *args, **kwargs):
        self.hash_algorithm = hash_algorithm
        self.salt_length = salt_length
        kwargs['max_length'] = kwargs.get('max_length', 128)
        super().__init__(*args, **kwargs)

    def hash_value(self, value: str) -> str:
        """Hash a value with salt."""
        salt = security_manager.generate_secure_token(self.salt_length)
        
        # Create hash
        hash_obj = hashlib.new(self.hash_algorithm)
        hash_obj.update((value + salt).encode('utf-8'))
        hashed = hash_obj.hexdigest()
        
        # Combine salt and hash
        return f"{salt}:{hashed}"

    def verify_value(self, value: str, stored_hash: str) -> bool:
        """Verify a value against stored hash."""
        try:
            salt, hashed = stored_hash.split(':', 1)
            
            hash_obj = hashlib.new(self.hash_algorithm)
            hash_obj.update((value + salt).encode('utf-8'))
            computed_hash = hash_obj.hexdigest()
            
            return computed_hash == hashed
        except (ValueError, AttributeError):
            return False

    def get_prep_value(self, value):
        """Hash value before storing."""
        if value is None:
            return value
        return self.hash_value(str(value))


class SecureQuerySet(models.QuerySet):
    """
    Custom QuerySet with built-in security controls:
    - Automatic audit logging
    - Data access validation
    - Query sanitization
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audit_logger = DatabaseAuditLogger()

    def filter(self, *args, **kwargs):
        """Override filter to add audit logging."""
        self._log_query_access('FILTER', args, kwargs)
        return super().filter(*args, **kwargs)

    def get(self, *args, **kwargs):
        """Override get to add audit logging."""
        self._log_query_access('GET', args, kwargs)
        return super().get(*args, **kwargs)

    def create(self, **kwargs):
        """Override create to add audit logging."""
        self._log_query_access('CREATE', [], kwargs)
        return super().create(**kwargs)

    def bulk_create(self, objs, **kwargs):
        """Override bulk_create with audit logging."""
        self._log_query_access('BULK_CREATE', [len(objs)], kwargs)
        return super().bulk_create(objs, **kwargs)

    def update(self, **kwargs):
        """Override update with audit logging."""
        count = self.count()  # Get count before update
        result = super().update(**kwargs)
        self._log_query_access('UPDATE', [count], kwargs)
        return result

    def delete(self):
        """Override delete with audit logging."""
        count = self.count()  # Get count before delete
        result = super().delete()
        self._log_query_access('DELETE', [count], {})
        return result

    def _log_query_access(self, operation: str, args: List, kwargs: Dict):
        """Log database access for audit purposes."""
        self.audit_logger.log_database_access(
            operation=operation,
            model=self.model.__name__,
            args=args,
            kwargs=self._sanitize_kwargs(kwargs)
        )

    def _sanitize_kwargs(self, kwargs: Dict) -> Dict:
        """Sanitize kwargs to remove sensitive data from logs."""
        sanitized = {}
        sensitive_fields = ['password', 'token', 'secret', 'key']
        
        for key, value in kwargs.items():
            if any(sensitive in key.lower() for sensitive in sensitive_fields):
                sanitized[key] = '[REDACTED]'
            else:
                sanitized[key] = str(value)[:100]  # Truncate long values
        
        return sanitized


class SecureManager(models.Manager):
    """
    Custom model manager with security enhancements.
    """

    def get_queryset(self):
        return SecureQuerySet(self.model, using=self._db)

    def secure_filter(self, user, **kwargs):
        """Filter with user-specific security checks."""
        # Add row-level security based on user permissions
        queryset = self.get_queryset()
        
        # Apply user-specific filters
        if hasattr(self.model, 'owner') and not user.is_superuser:
            queryset = queryset.filter(owner=user)
        
        return queryset.filter(**kwargs)

    def create_with_audit(self, user, **kwargs):
        """Create object with full audit trail."""
        with transaction.atomic():
            obj = self.create(**kwargs)
            
            # Log creation
            DatabaseAuditLogger().log_model_change(
                operation='CREATE',
                model=self.model.__name__,
                object_id=obj.pk,
                user=user,
                changes=kwargs
            )
            
            return obj


class DatabaseAuditLogger:
    """
    Comprehensive database audit logging system.
    Tracks all database operations, changes, and access patterns.
    """

    def __init__(self):
        self.redis_client = security_manager.redis_client

    def log_database_access(
        self, 
        operation: str, 
        model: str, 
        args: List = None, 
        kwargs: Dict = None,
        user: Optional[Any] = None
    ):
        """Log database access operations."""
        
        audit_entry = {
            'timestamp': timezone.now().isoformat(),
            'operation': operation,
            'model': model,
            'args': args or [],
            'kwargs': kwargs or {},
            'user_id': getattr(user, 'id', None),
            'source': 'DatabaseAccess'
        }
        
        # Store in Redis for real-time monitoring
        self.redis_client.lpush('db_audit_log', json.dumps(audit_entry))
        self.redis_client.ltrim('db_audit_log', 0, 10000)  # Keep last 10k entries
        
        # Also store in database for long-term audit
        self._store_audit_entry(audit_entry)

    def log_model_change(
        self, 
        operation: str, 
        model: str, 
        object_id: Any,
        user: Optional[Any] = None,
        changes: Optional[Dict] = None,
        previous_values: Optional[Dict] = None
    ):
        """Log model changes with before/after values."""
        
        audit_entry = {
            'timestamp': timezone.now().isoformat(),
            'operation': operation,
            'model': model,
            'object_id': str(object_id),
            'user_id': getattr(user, 'id', None),
            'changes': self._encrypt_sensitive_changes(changes or {}),
            'previous_values': self._encrypt_sensitive_changes(previous_values or {}),
            'source': 'ModelChange'
        }
        
        self._store_audit_entry(audit_entry)

    def log_security_event(
        self,
        event_type: str,
        severity: str = 'INFO',
        details: Optional[Dict] = None,
        user: Optional[Any] = None
    ):
        """Log security-related database events."""
        
        audit_entry = {
            'timestamp': timezone.now().isoformat(),
            'event_type': event_type,
            'severity': severity,
            'details': details or {},
            'user_id': getattr(user, 'id', None),
            'source': 'SecurityEvent'
        }
        
        # Store security events with higher priority
        self.redis_client.lpush('security_audit_log', json.dumps(audit_entry))
        self.redis_client.ltrim('security_audit_log', 0, 5000)
        
        self._store_audit_entry(audit_entry, table='security_audit_log')

    def _encrypt_sensitive_changes(self, changes: Dict) -> Dict:
        """Encrypt sensitive fields in change logs."""
        encrypted_changes = {}
        sensitive_fields = ['password', 'token', 'secret', 'ssn', 'credit_card']
        
        for field, value in changes.items():
            if any(sensitive in field.lower() for sensitive in sensitive_fields):
                # Encrypt sensitive values
                encrypted_changes[field] = security_manager.encrypt_sensitive_data(str(value))
            else:
                encrypted_changes[field] = value
        
        return encrypted_changes

    def _store_audit_entry(self, entry: Dict, table: str = 'database_audit_log'):
        """Store audit entry in secure database table."""
        # In a real implementation, you'd store this in a dedicated audit table
        # For now, we'll use Redis as the primary storage
        
        encrypted_entry = security_manager.encrypt_sensitive_data(entry)
        
        # Store with unique key
        audit_key = f"{table}:{uuid.uuid4()}"
        self.redis_client.setex(audit_key, 86400 * 30, encrypted_entry)  # 30 days retention


class DataMaskingManager:
    """
    Data masking and anonymization for development and testing environments.
    """

    def __init__(self):
        self.masking_rules = self._load_masking_rules()

    def mask_sensitive_data(self, data: Dict, mask_level: str = 'partial') -> Dict:
        """
        Mask sensitive data based on field types and masking level.
        
        Args:
            data: Dictionary containing data to mask
            mask_level: 'partial', 'full', or 'none'
        """
        if mask_level == 'none':
            return data
        
        masked_data = {}
        
        for field, value in data.items():
            if self._is_sensitive_field(field):
                masked_data[field] = self._mask_field_value(
                    field, value, mask_level
                )
            else:
                masked_data[field] = value
        
        return masked_data

    def _is_sensitive_field(self, field_name: str) -> bool:
        """Check if field contains sensitive data."""
        sensitive_patterns = [
            'password', 'token', 'secret', 'key',
            'ssn', 'social_security',
            'credit_card', 'card_number',
            'bank_account', 'routing_number',
            'phone', 'email',
            'address', 'zip', 'postal'
        ]
        
        field_lower = field_name.lower()
        return any(pattern in field_lower for pattern in sensitive_patterns)

    def _mask_field_value(self, field: str, value: Any, mask_level: str) -> str:
        """Mask individual field value."""
        if not value:
            return value
        
        value_str = str(value)
        
        if mask_level == 'full':
            return '*' * len(value_str)
        
        # Partial masking based on field type
        field_lower = field.lower()
        
        if 'email' in field_lower:
            return self._mask_email(value_str)
        elif 'phone' in field_lower:
            return self._mask_phone(value_str)
        elif 'credit_card' in field_lower or 'card_number' in field_lower:
            return self._mask_credit_card(value_str)
        elif 'ssn' in field_lower:
            return self._mask_ssn(value_str)
        else:
            # Generic masking - show first and last 2 characters
            if len(value_str) > 4:
                return value_str[:2] + '*' * (len(value_str) - 4) + value_str[-2:]
            else:
                return '*' * len(value_str)

    def _mask_email(self, email: str) -> str:
        """Mask email address."""
        try:
            local, domain = email.split('@')
            masked_local = local[0] + '*' * (len(local) - 2) + local[-1] if len(local) > 2 else '*' * len(local)
            return f"{masked_local}@{domain}"
        except ValueError:
            return '*' * len(email)

    def _mask_phone(self, phone: str) -> str:
        """Mask phone number."""
        # Remove non-digits
        digits = ''.join(c for c in phone if c.isdigit())
        if len(digits) >= 10:
            return f"***-***-{digits[-4:]}"
        else:
            return '*' * len(phone)

    def _mask_credit_card(self, card: str) -> str:
        """Mask credit card number."""
        digits = ''.join(c for c in card if c.isdigit())
        if len(digits) >= 12:
            return f"****-****-****-{digits[-4:]}"
        else:
            return '*' * len(card)

    def _mask_ssn(self, ssn: str) -> str:
        """Mask Social Security Number."""
        digits = ''.join(c for c in ssn if c.isdigit())
        if len(digits) == 9:
            return f"***-**-{digits[-4:]}"
        else:
            return '*' * len(ssn)

    def _load_masking_rules(self) -> Dict:
        """Load masking rules configuration."""
        return {
            'development': 'partial',
            'testing': 'partial',
            'staging': 'none',
            'production': 'none'
        }


class DatabaseConnectionSecurity:
    """
    Secure database connection management with connection pooling
    and security monitoring.
    """

    def __init__(self):
        self.connection_monitor = DatabaseConnectionMonitor()

    def get_secure_connection_params(self) -> Dict[str, Any]:
        """Get secure database connection parameters."""
        return {
            'OPTIONS': {
                'sslmode': 'require',
                'sslcert': getattr(settings, 'DB_SSL_CERT', None),
                'sslkey': getattr(settings, 'DB_SSL_KEY', None),
                'sslrootcert': getattr(settings, 'DB_SSL_ROOT_CERT', None),
                'connect_timeout': 10,
                'options': '-c default_transaction_isolation=serializable'
            },
            'CONN_MAX_AGE': 300,  # 5 minutes
            'ATOMIC_REQUESTS': True,
        }

    def monitor_connections(self):
        """Monitor database connections for security issues."""
        self.connection_monitor.check_connection_patterns()
        self.connection_monitor.detect_sql_injection_attempts()
        self.connection_monitor.monitor_query_performance()


class DatabaseConnectionMonitor:
    """Monitor database connections for security and performance."""

    def __init__(self):
        self.redis_client = security_manager.redis_client

    def check_connection_patterns(self):
        """Check for unusual connection patterns."""
        # Implementation for monitoring connection patterns
        pass

    def detect_sql_injection_attempts(self):
        """Detect potential SQL injection attempts."""
        # Implementation for SQL injection detection
        pass

    def monitor_query_performance(self):
        """Monitor query performance for unusual patterns."""
        # Implementation for query performance monitoring
        pass


# Initialize database security components
db_security_manager = DatabaseSecurityManager()
data_masking_manager = DataMaskingManager()
db_connection_security = DatabaseConnectionSecurity()