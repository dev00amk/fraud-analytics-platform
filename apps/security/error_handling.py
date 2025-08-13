"""
Enterprise Error Handling and Security Response System
Comprehensive error management, incident response, and security event handling
"""

import sys
import traceback
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from django.conf import settings
from django.core.exceptions import ValidationError, PermissionDenied
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.generic import View

from .audit_monitoring import audit_logger, SecurityEventType, SecurityEventSeverity
from .security_manager import security_manager


class ErrorSeverity(Enum):
    """Error severity levels for proper escalation."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ErrorCategory(Enum):
    """Categories of errors for proper handling."""
    AUTHENTICATION = "AUTHENTICATION"
    AUTHORIZATION = "AUTHORIZATION"
    VALIDATION = "VALIDATION"
    SECURITY = "SECURITY"
    SYSTEM = "SYSTEM"
    BUSINESS = "BUSINESS"
    EXTERNAL = "EXTERNAL"


class SecurityError(Exception):
    """Base class for security-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SECURITY,
        details: Dict = None
    ):
        self.message = message
        self.error_code = error_code or f"SEC_{int(datetime.now().timestamp())}"
        self.severity = severity
        self.category = category
        self.details = details or {}
        self.timestamp = timezone.now()
        super().__init__(message)


class AuthenticationError(SecurityError):
    """Authentication-related errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.AUTHENTICATION)
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)


class AuthorizationError(SecurityError):
    """Authorization-related errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.AUTHORIZATION)
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)


class ValidationError(SecurityError):
    """Validation-related errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.VALIDATION)
        kwargs.setdefault('severity', ErrorSeverity.LOW)
        super().__init__(message, **kwargs)


class SecurityViolationError(SecurityError):
    """Security violation errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.SECURITY)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class SystemError(SecurityError):
    """System-level errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.SYSTEM)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class ErrorHandler:
    """
    Comprehensive error handler with security-focused error management.
    Provides structured error responses while preventing information disclosure.
    """

    def __init__(self):
        self.error_patterns = self._load_error_patterns()
        self.response_templates = self._load_response_templates()
        self.escalation_rules = self._load_escalation_rules()

    def handle_error(
        self,
        error: Exception,
        request: HttpRequest = None,
        context: Dict = None
    ) -> HttpResponse:
        """
        Handle error with appropriate logging, alerting, and response generation.
        
        Args:
            error: The exception that occurred
            request: HTTP request object (if applicable)
            context: Additional context information
            
        Returns:
            Appropriate HTTP response
        """
        
        # Generate unique error ID for tracking
        error_id = self._generate_error_id()
        
        # Classify error
        error_info = self._classify_error(error, request, context)
        
        # Log error with full context
        self._log_error(error_id, error, error_info, request, context)
        
        # Check for security implications
        if self._is_security_related(error, error_info):
            self._handle_security_error(error_id, error, error_info, request)
        
        # Escalate if needed
        if self._should_escalate(error_info):
            self._escalate_error(error_id, error, error_info, request)
        
        # Generate appropriate response
        return self._generate_error_response(error_id, error, error_info, request)

    def _generate_error_id(self) -> str:
        """Generate unique error ID for tracking."""
        return f"ERR_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:8]}"

    def _classify_error(
        self,
        error: Exception,
        request: HttpRequest = None,
        context: Dict = None
    ) -> Dict[str, Any]:
        """Classify error to determine appropriate handling."""
        
        error_info = {
            'type': type(error).__name__,
            'message': str(error),
            'severity': ErrorSeverity.LOW,
            'category': ErrorCategory.SYSTEM,
            'is_user_error': False,
            'is_security_related': False,
            'should_expose': False,
            'http_status': 500
        }
        
        # Handle custom security errors
        if isinstance(error, SecurityError):
            error_info.update({
                'severity': error.severity,
                'category': error.category,
                'error_code': error.error_code,
                'is_security_related': True,
                'should_expose': True,
                'details': error.details
            })
            
            # Set appropriate HTTP status
            if error.category == ErrorCategory.AUTHENTICATION:
                error_info['http_status'] = 401
            elif error.category == ErrorCategory.AUTHORIZATION:
                error_info['http_status'] = 403
            elif error.category == ErrorCategory.VALIDATION:
                error_info['http_status'] = 400
                error_info['is_user_error'] = True
            else:
                error_info['http_status'] = 500
        
        # Handle Django built-in exceptions
        elif isinstance(error, PermissionDenied):
            error_info.update({
                'severity': ErrorSeverity.MEDIUM,
                'category': ErrorCategory.AUTHORIZATION,
                'is_security_related': True,
                'should_expose': True,
                'http_status': 403
            })
        
        elif isinstance(error, ValidationError):
            error_info.update({
                'severity': ErrorSeverity.LOW,
                'category': ErrorCategory.VALIDATION,
                'is_user_error': True,
                'should_expose': True,
                'http_status': 400
            })
        
        # Pattern-based classification
        else:
            self._apply_pattern_classification(error, error_info)
        
        return error_info

    def _apply_pattern_classification(self, error: Exception, error_info: Dict):
        """Apply pattern-based error classification."""
        
        error_message = str(error).lower()
        
        for pattern_info in self.error_patterns:
            if any(keyword in error_message for keyword in pattern_info['keywords']):
                error_info.update({
                    'severity': ErrorSeverity(pattern_info['severity']),
                    'category': ErrorCategory(pattern_info['category']),
                    'is_security_related': pattern_info.get('security_related', False),
                    'should_expose': pattern_info.get('expose', False),
                    'http_status': pattern_info.get('http_status', 500)
                })
                break

    def _is_security_related(self, error: Exception, error_info: Dict) -> bool:
        """Determine if error is security-related."""
        return error_info.get('is_security_related', False)

    def _should_escalate(self, error_info: Dict) -> bool:
        """Determine if error should be escalated."""
        severity = error_info.get('severity', ErrorSeverity.LOW)
        return severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]

    def _log_error(
        self,
        error_id: str,
        error: Exception,
        error_info: Dict,
        request: HttpRequest = None,
        context: Dict = None
    ):
        """Log error with comprehensive information."""
        
        # Prepare log data
        log_data = {
            'error_id': error_id,
            'error_type': error_info['type'],
            'error_message': error_info['message'],
            'severity': error_info['severity'].value,
            'category': error_info['category'].value,
            'traceback': self._get_safe_traceback(error),
            'context': context or {}
        }
        
        # Add request information if available
        if request:
            log_data.update({
                'request_method': request.method,
                'request_path': request.path,
                'request_user': getattr(request.user, 'id', None) if hasattr(request, 'user') else None,
                'remote_addr': self._get_client_ip(request),
                'user_agent': request.META.get('HTTP_USER_AGENT', '')[:500]
            })
        
        # Log to audit system
        audit_logger.log_security_event(
            event_type=SecurityEventType.SECURITY_VIOLATION if error_info['is_security_related'] else SecurityEventType.SYSTEM_CONFIGURATION_CHANGE,
            severity=self._map_to_security_severity(error_info['severity']),
            message=f"Error occurred: {error_info['type']}",
            details=log_data
        )

    def _handle_security_error(
        self,
        error_id: str,
        error: Exception,
        error_info: Dict,
        request: HttpRequest = None
    ):
        """Handle security-related errors with additional measures."""
        
        # Log security event
        audit_logger.log_security_event(
            event_type=SecurityEventType.SECURITY_VIOLATION,
            severity=self._map_to_security_severity(error_info['severity']),
            message=f"Security error: {error_info['type']}",
            user_id=getattr(request.user, 'id', None) if request and hasattr(request, 'user') else None,
            ip_address=self._get_client_ip(request) if request else None,
            user_agent=request.META.get('HTTP_USER_AGENT', '') if request else None,
            details={
                'error_id': error_id,
                'error_type': error_info['type'],
                'error_message': error_info['message'],
                'severity': error_info['severity'].value
            }
        )
        
        # Additional security measures based on severity
        if error_info['severity'] == ErrorSeverity.CRITICAL:
            self._handle_critical_security_error(error_id, error, error_info, request)
        elif error_info['severity'] == ErrorSeverity.HIGH:
            self._handle_high_security_error(error_id, error, error_info, request)

    def _handle_critical_security_error(
        self,
        error_id: str,
        error: Exception,
        error_info: Dict,
        request: HttpRequest = None
    ):
        """Handle critical security errors with immediate response."""
        
        # Immediate notification to security team
        self._send_security_alert(
            severity='CRITICAL',
            title=f'Critical Security Error: {error_info["type"]}',
            message=f'Error ID: {error_id}\nType: {error_info["type"]}\nMessage: {error_info["message"]}',
            error_id=error_id
        )
        
        # Consider temporary measures
        if request:
            client_ip = self._get_client_ip(request)
            
            # Temporary IP blocking for repeated critical errors
            self._consider_temporary_ip_block(client_ip, error_info)

    def _handle_high_security_error(
        self,
        error_id: str,
        error: Exception,
        error_info: Dict,
        request: HttpRequest = None
    ):
        """Handle high severity security errors."""
        
        # Send alert to security team
        self._send_security_alert(
            severity='HIGH',
            title=f'High Severity Security Error: {error_info["type"]}',
            message=f'Error ID: {error_id}\nType: {error_info["type"]}\nMessage: {error_info["message"]}',
            error_id=error_id
        )

    def _escalate_error(
        self,
        error_id: str,
        error: Exception,
        error_info: Dict,
        request: HttpRequest = None
    ):
        """Escalate error according to escalation rules."""
        
        escalation_info = {
            'error_id': error_id,
            'error_type': error_info['type'],
            'severity': error_info['severity'].value,
            'category': error_info['category'].value,
            'timestamp': timezone.now().isoformat()
        }
        
        # Add to escalation queue
        self._add_to_escalation_queue(escalation_info)

    def _generate_error_response(
        self,
        error_id: str,
        error: Exception,
        error_info: Dict,
        request: HttpRequest = None
    ) -> HttpResponse:
        """Generate appropriate error response."""
        
        # Determine response content
        if error_info.get('should_expose', False):
            # Safe to expose error details
            error_message = error_info['message']
            error_code = error_info.get('error_code')
        else:
            # Generic error message to prevent information disclosure
            error_message = self._get_generic_error_message(error_info)
            error_code = None
        
        # Prepare response data
        response_data = {
            'error': True,
            'message': error_message,
            'error_id': error_id,
            'timestamp': timezone.now().isoformat()
        }
        
        if error_code:
            response_data['error_code'] = error_code
        
        # Add validation details for user errors
        if error_info.get('is_user_error', False) and hasattr(error, 'error_dict'):
            response_data['details'] = error.error_dict
        
        return JsonResponse(
            response_data,
            status=error_info.get('http_status', 500)
        )

    def _get_safe_traceback(self, error: Exception) -> str:
        """Get sanitized traceback that doesn't expose sensitive information."""
        
        if settings.DEBUG:
            # Full traceback in debug mode
            return traceback.format_exc()
        else:
            # Sanitized traceback for production
            tb_lines = traceback.format_exc().split('\n')
            
            # Filter out sensitive paths
            safe_lines = []
            for line in tb_lines:
                if any(sensitive in line.lower() for sensitive in ['password', 'secret', 'key', 'token']):
                    safe_lines.append('[SENSITIVE LINE REMOVED]')
                else:
                    # Remove absolute paths
                    if 'File "' in line:
                        parts = line.split('File "')
                        if len(parts) > 1:
                            file_path = parts[1].split('"')[0]
                            filename = file_path.split('/')[-1] if '/' in file_path else file_path.split('\\')[-1]
                            line = parts[0] + f'File "{filename}"' + '"'.join(parts[1].split('"')[1:])
                    safe_lines.append(line)
            
            return '\n'.join(safe_lines)

    def _get_client_ip(self, request: HttpRequest) -> str:
        """Get client IP address from request."""
        
        forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.META.get('HTTP_X_REAL_IP')
        if real_ip:
            return real_ip
        
        return request.META.get('REMOTE_ADDR', '0.0.0.0')

    def _get_generic_error_message(self, error_info: Dict) -> str:
        """Get generic error message based on category."""
        
        messages = {
            ErrorCategory.AUTHENTICATION: "Authentication failed. Please check your credentials.",
            ErrorCategory.AUTHORIZATION: "You don't have permission to perform this action.",
            ErrorCategory.VALIDATION: "Invalid input data provided.",
            ErrorCategory.SECURITY: "Security validation failed.",
            ErrorCategory.SYSTEM: "A system error occurred. Please try again later.",
            ErrorCategory.BUSINESS: "Business rule validation failed.",
            ErrorCategory.EXTERNAL: "External service error. Please try again later."
        }
        
        return messages.get(error_info['category'], "An error occurred. Please try again later.")

    def _map_to_security_severity(self, error_severity: ErrorSeverity) -> SecurityEventSeverity:
        """Map error severity to security event severity."""
        
        mapping = {
            ErrorSeverity.LOW: SecurityEventSeverity.LOW,
            ErrorSeverity.MEDIUM: SecurityEventSeverity.MEDIUM,
            ErrorSeverity.HIGH: SecurityEventSeverity.HIGH,
            ErrorSeverity.CRITICAL: SecurityEventSeverity.CRITICAL
        }
        
        return mapping.get(error_severity, SecurityEventSeverity.LOW)

    def _send_security_alert(self, severity: str, title: str, message: str, error_id: str):
        """Send security alert to appropriate channels."""
        # Implementation would integrate with alerting system
        pass

    def _consider_temporary_ip_block(self, ip_address: str, error_info: Dict):
        """Consider temporary IP blocking for repeated critical errors."""
        
        # Check error frequency from this IP
        recent_errors_key = f"critical_errors:{ip_address}"
        error_count = security_manager.redis_client.incr(recent_errors_key)
        security_manager.redis_client.expire(recent_errors_key, 300)  # 5 minutes
        
        # Block IP if too many critical errors
        if error_count >= 3:
            block_key = f"blocked_ip:{ip_address}"
            security_manager.redis_client.setex(block_key, 3600, 1)  # 1 hour block
            
            audit_logger.log_security_event(
                event_type=SecurityEventType.SECURITY_VIOLATION,
                severity=SecurityEventSeverity.CRITICAL,
                message=f"IP temporarily blocked due to repeated critical errors: {ip_address}",
                ip_address=ip_address,
                details={'error_count': error_count, 'block_duration': 3600}
            )

    def _add_to_escalation_queue(self, escalation_info: Dict):
        """Add error to escalation queue."""
        
        queue_key = 'error_escalation_queue'
        security_manager.redis_client.lpush(queue_key, str(escalation_info))
        security_manager.redis_client.ltrim(queue_key, 0, 1000)  # Keep last 1000

    def _load_error_patterns(self) -> List[Dict]:
        """Load error classification patterns."""
        
        return [
            {
                'keywords': ['authentication', 'login', 'credentials', 'password'],
                'severity': 'MEDIUM',
                'category': 'AUTHENTICATION',
                'security_related': True,
                'expose': True,
                'http_status': 401
            },
            {
                'keywords': ['permission', 'authorization', 'access denied', 'forbidden'],
                'severity': 'MEDIUM',
                'category': 'AUTHORIZATION',
                'security_related': True,
                'expose': True,
                'http_status': 403
            },
            {
                'keywords': ['sql injection', 'xss', 'csrf', 'malicious'],
                'severity': 'HIGH',
                'category': 'SECURITY',
                'security_related': True,
                'expose': False,
                'http_status': 400
            },
            {
                'keywords': ['database', 'connection', 'timeout'],
                'severity': 'HIGH',
                'category': 'SYSTEM',
                'security_related': False,
                'expose': False,
                'http_status': 500
            },
            {
                'keywords': ['validation', 'invalid', 'required field'],
                'severity': 'LOW',
                'category': 'VALIDATION',
                'security_related': False,
                'expose': True,
                'http_status': 400
            }
        ]

    def _load_response_templates(self) -> Dict[str, str]:
        """Load error response templates."""
        
        return {
            'generic_error': "An error occurred. Please try again later.",
            'authentication_error': "Authentication failed. Please check your credentials.",
            'authorization_error': "You don't have permission to perform this action.",
            'validation_error': "Invalid input data provided.",
            'security_error': "Security validation failed.",
            'system_error': "A system error occurred. Please contact support.",
        }

    def _load_escalation_rules(self) -> Dict[str, Any]:
        """Load error escalation rules."""
        
        return {
            'critical_immediate': {
                'severity': [ErrorSeverity.CRITICAL],
                'channels': ['email', 'sms', 'slack'],
                'recipients': ['security-team', 'on-call'],
                'delay': 0
            },
            'high_priority': {
                'severity': [ErrorSeverity.HIGH],
                'channels': ['email', 'slack'],
                'recipients': ['security-team', 'dev-team'],
                'delay': 300  # 5 minutes
            },
            'medium_priority': {
                'severity': [ErrorSeverity.MEDIUM],
                'channels': ['email'],
                'recipients': ['dev-team'],
                'delay': 3600  # 1 hour
            }
        }


class SecurityIncidentResponse:
    """
    Automated security incident response system.
    """

    def __init__(self):
        self.response_playbooks = self._load_response_playbooks()
        self.incident_counter = 0

    def handle_security_incident(
        self,
        incident_type: str,
        severity: SecurityEventSeverity,
        details: Dict,
        source: str = None
    ) -> str:
        """Handle security incident with automated response."""
        
        incident_id = self._generate_incident_id()
        
        # Log incident
        audit_logger.log_security_event(
            event_type=SecurityEventType.SECURITY_VIOLATION,
            severity=severity,
            message=f"Security incident: {incident_type}",
            details={
                'incident_id': incident_id,
                'incident_type': incident_type,
                'source': source,
                **details
            }
        )
        
        # Execute response playbook
        playbook = self.response_playbooks.get(incident_type, {})
        if playbook:
            self._execute_response_playbook(incident_id, playbook, details)
        
        return incident_id

    def _generate_incident_id(self) -> str:
        """Generate unique incident ID."""
        self.incident_counter += 1
        timestamp = int(datetime.now().timestamp())
        return f"INC_{timestamp}_{self.incident_counter:04d}"

    def _execute_response_playbook(
        self,
        incident_id: str,
        playbook: Dict,
        details: Dict
    ):
        """Execute automated response playbook."""
        
        for action in playbook.get('actions', []):
            try:
                self._execute_response_action(incident_id, action, details)
            except Exception as e:
                # Log action failure but continue with other actions
                audit_logger.log_security_event(
                    event_type=SecurityEventType.SYSTEM_CONFIGURATION_CHANGE,
                    severity=SecurityEventSeverity.MEDIUM,
                    message=f"Response action failed for incident {incident_id}",
                    details={
                        'incident_id': incident_id,
                        'action': action,
                        'error': str(e)
                    }
                )

    def _execute_response_action(self, incident_id: str, action: Dict, details: Dict):
        """Execute individual response action."""
        
        action_type = action.get('type')
        
        if action_type == 'block_ip':
            self._block_ip_address(details.get('ip_address'), action.get('duration', 3600))
        elif action_type == 'disable_user':
            self._disable_user_account(details.get('user_id'), action.get('duration'))
        elif action_type == 'alert':
            self._send_security_alert(action, incident_id)
        elif action_type == 'isolate_session':
            self._isolate_user_session(details.get('session_id'))

    def _block_ip_address(self, ip_address: str, duration: int):
        """Block IP address temporarily."""
        if ip_address:
            block_key = f"blocked_ip:{ip_address}"
            security_manager.redis_client.setex(block_key, duration, 1)

    def _disable_user_account(self, user_id: str, duration: int = None):
        """Temporarily disable user account."""
        if user_id:
            disable_key = f"disabled_user:{user_id}"
            security_manager.redis_client.setex(disable_key, duration or 3600, 1)

    def _send_security_alert(self, action: Dict, incident_id: str):
        """Send security alert."""
        # Implementation would integrate with alerting system
        pass

    def _isolate_user_session(self, session_id: str):
        """Isolate user session."""
        if session_id:
            isolation_key = f"isolated_session:{session_id}"
            security_manager.redis_client.setex(isolation_key, 3600, 1)

    def _load_response_playbooks(self) -> Dict[str, Dict]:
        """Load incident response playbooks."""
        
        return {
            'brute_force_attack': {
                'actions': [
                    {'type': 'block_ip', 'duration': 3600},
                    {'type': 'alert', 'severity': 'HIGH', 'channels': ['email', 'slack']}
                ]
            },
            'sql_injection_attempt': {
                'actions': [
                    {'type': 'block_ip', 'duration': 7200},
                    {'type': 'alert', 'severity': 'CRITICAL', 'channels': ['email', 'sms', 'slack']},
                    {'type': 'isolate_session'}
                ]
            },
            'suspicious_activity': {
                'actions': [
                    {'type': 'alert', 'severity': 'MEDIUM', 'channels': ['email']},
                    {'type': 'isolate_session'}
                ]
            },
            'fraud_detection': {
                'actions': [
                    {'type': 'alert', 'severity': 'HIGH', 'channels': ['email', 'slack']},
                    {'type': 'disable_user', 'duration': 1800}  # 30 minutes
                ]
            }
        }


# Initialize error handling components
error_handler = ErrorHandler()
incident_response = SecurityIncidentResponse()


# Django middleware for global error handling
class SecurityErrorMiddleware:
    """Django middleware for comprehensive error handling."""
    
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        try:
            response = self.get_response(request)
            return response
        except Exception as error:
            return error_handler.handle_error(error, request)

    def process_exception(self, request, exception):
        """Process unhandled exceptions."""
        return error_handler.handle_error(exception, request)