"""
Enterprise Security Framework for Fraud Analytics Platform

This module provides a comprehensive security framework including:
- Authentication and authorization
- API security and rate limiting
- Database encryption and security
- Audit logging and monitoring
- Input validation and sanitization
- Configuration management
- Error handling and incident response

Usage:
    from apps.security import (
        security_manager,
        auth_manager,
        authz_manager,
        security_monitor,
        audit_logger,
        input_validator,
        config_manager
    )
"""

from .security_manager import (
    SecurityManager,
    AuthenticationManager,
    AuthorizationManager,
    SecurityMonitor,
    security_manager,
    auth_manager,
    authz_manager,
    security_monitor
)

from .api_security import (
    APISecurityMiddleware,
    RateLimiter,
    DDoSProtector,
    RequestValidator,
    SecurityHeaders
)

from .database_security import (
    DatabaseSecurityManager,
    EncryptedField,
    HashedField,
    SecureQuerySet,
    SecureManager,
    DatabaseAuditLogger,
    DataMaskingManager,
    db_security_manager,
    data_masking_manager
)

from .audit_monitoring import (
    AuditLogger,
    SecurityMetricsCollector,
    ComplianceMonitor,
    SecurityAlertManager,
    SecurityEventType,
    SecurityEventSeverity,
    audit_logger,
    security_metrics,
    compliance_monitor,
    alert_manager
)

from .input_validation import (
    InputValidator,
    TransactionValidator,
    APIRequestValidator,
    input_validator,
    transaction_validator,
    api_request_validator
)

from .config_management import (
    SecureConfigManager,
    SecretsManager,
    EnvironmentManager,
    config_manager,
    secrets_manager,
    environment_manager
)

from .error_handling import (
    ErrorHandler,
    SecurityIncidentResponse,
    SecurityErrorMiddleware,
    SecurityError,
    AuthenticationError,
    AuthorizationError,
    ValidationError,
    SecurityViolationError,
    SystemError,
    ErrorSeverity,
    ErrorCategory,
    error_handler,
    incident_response
)

# Version information
__version__ = '1.0.0'
__author__ = 'Fraud Analytics Security Team'

# Export all main components
__all__ = [
    # Core security managers
    'security_manager',
    'auth_manager',
    'authz_manager',
    'security_monitor',
    
    # API security components
    'APISecurityMiddleware',
    'RateLimiter',
    'DDoSProtector',
    'RequestValidator',
    'SecurityHeaders',
    
    # Database security
    'EncryptedField',
    'HashedField',
    'SecureQuerySet',
    'SecureManager',
    'db_security_manager',
    'data_masking_manager',
    
    # Audit and monitoring
    'audit_logger',
    'security_metrics',
    'compliance_monitor',
    'alert_manager',
    'SecurityEventType',
    'SecurityEventSeverity',
    
    # Input validation
    'input_validator',
    'transaction_validator',
    'api_request_validator',
    
    # Configuration management
    'config_manager',
    'secrets_manager',
    'environment_manager',
    
    # Error handling
    'error_handler',
    'incident_response',
    'SecurityErrorMiddleware',
    'SecurityError',
    'AuthenticationError',
    'AuthorizationError',
    'ValidationError',
    'SecurityViolationError',
    'SystemError',
    'ErrorSeverity',
    'ErrorCategory',
    
    # Classes for custom implementations
    'SecurityManager',
    'AuthenticationManager',
    'AuthorizationManager',
    'SecurityMonitor',
    'DatabaseSecurityManager',
    'EncryptedField',
    'HashedField',
    'AuditLogger',
    'SecurityMetricsCollector',
    'ComplianceMonitor',
    'SecurityAlertManager',
    'InputValidator',
    'TransactionValidator',
    'APIRequestValidator',
    'SecureConfigManager',
    'SecretsManager',
    'EnvironmentManager',
    'ErrorHandler',
    'SecurityIncidentResponse'
]