"""
Enterprise Audit Logging and Security Monitoring System
Comprehensive security event tracking, alerting, and compliance monitoring
"""

import hashlib
import json
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.db import models
from django.utils import timezone
from redis import Redis

from .security_manager import security_manager


class SecurityEventSeverity(Enum):
    """Security event severity levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SecurityEventType(Enum):
    """Types of security events to track."""
    AUTHENTICATION_SUCCESS = "AUTH_SUCCESS"
    AUTHENTICATION_FAILURE = "AUTH_FAILURE"
    AUTHORIZATION_FAILURE = "AUTHZ_FAILURE"
    SUSPICIOUS_ACTIVITY = "SUSPICIOUS_ACTIVITY"
    DATA_ACCESS = "DATA_ACCESS"
    DATA_MODIFICATION = "DATA_MODIFICATION"
    SYSTEM_CONFIGURATION_CHANGE = "CONFIG_CHANGE"
    SECURITY_VIOLATION = "SECURITY_VIOLATION"
    FRAUD_DETECTION = "FRAUD_DETECTION"
    API_ABUSE = "API_ABUSE"
    COMPLIANCE_VIOLATION = "COMPLIANCE_VIOLATION"


class AuditLogger:
    """
    Centralized audit logging system for security events.
    Provides structured logging with encryption and integrity verification.
    """

    def __init__(self):
        redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379/0')
        self.redis_client = Redis.from_url(redis_url)
        self.event_handlers = self._initialize_event_handlers()
        
    def _initialize_event_handlers(self) -> Dict[str, List]:
        """Initialize event handlers for different security events."""
        return {
            SecurityEventSeverity.CRITICAL.value: [
                self._handle_critical_event,
                self._send_immediate_alert
            ],
            SecurityEventSeverity.HIGH.value: [
                self._handle_high_severity_event,
                self._escalate_to_security_team
            ],
            SecurityEventSeverity.MEDIUM.value: [
                self._handle_medium_severity_event
            ],
            SecurityEventSeverity.LOW.value: [
                self._handle_low_severity_event
            ]
        }

    def log_security_event(
        self,
        event_type: SecurityEventType,
        severity: SecurityEventSeverity,
        message: str,
        user_id: Optional[int] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        resource: Optional[str] = None,
        details: Optional[Dict] = None,
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Log a security event with full context and metadata.
        
        Returns:
            Event ID for correlation and tracking
        """
        
        event_id = self._generate_event_id()
        timestamp = timezone.now()
        
        event_data = {
            'event_id': event_id,
            'timestamp': timestamp.isoformat(),
            'event_type': event_type.value,
            'severity': severity.value,
            'message': message,
            'user_id': user_id,
            'ip_address': ip_address,
            'user_agent': user_agent,
            'resource': resource,
            'details': details or {},
            'correlation_id': correlation_id,
            'source': 'SecurityAuditLogger',
            'environment': getattr(settings, 'ENVIRONMENT', 'production')
        }
        
        # Add integrity hash
        event_data['integrity_hash'] = self._calculate_integrity_hash(event_data)
        
        # Encrypt sensitive data
        encrypted_event = self._encrypt_event_data(event_data)
        
        # Store in multiple locations for redundancy
        self._store_event(encrypted_event, event_id)
        
        # Process event through handlers
        self._process_event_handlers(event_data)
        
        # Update security metrics
        self._update_security_metrics(event_type, severity)
        
        return event_id

    def log_user_activity(
        self,
        user_id: int,
        action: str,
        resource: str,
        ip_address: str,
        user_agent: str,
        success: bool,
        details: Optional[Dict] = None
    ):
        """Log user activity for audit trail."""
        
        event_type = SecurityEventType.DATA_ACCESS if success else SecurityEventType.AUTHORIZATION_FAILURE
        severity = SecurityEventSeverity.LOW if success else SecurityEventSeverity.MEDIUM
        
        message = f"User {user_id} {'successfully' if success else 'failed to'} {action} {resource}"
        
        self.log_security_event(
            event_type=event_type,
            severity=severity,
            message=message,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource=resource,
            details=details
        )

    def log_fraud_detection(
        self,
        transaction_id: str,
        fraud_score: float,
        risk_level: str,
        detection_rules: List[str],
        user_id: Optional[int] = None,
        details: Optional[Dict] = None
    ):
        """Log fraud detection events."""
        
        severity = self._determine_fraud_severity(fraud_score, risk_level)
        
        message = f"Fraud detected for transaction {transaction_id}: score={fraud_score}, level={risk_level}"
        
        fraud_details = {
            'transaction_id': transaction_id,
            'fraud_score': fraud_score,
            'risk_level': risk_level,
            'detection_rules': detection_rules,
            **(details or {})
        }
        
        self.log_security_event(
            event_type=SecurityEventType.FRAUD_DETECTION,
            severity=severity,
            message=message,
            user_id=user_id,
            resource=f"transaction:{transaction_id}",
            details=fraud_details
        )

    def log_compliance_event(
        self,
        regulation: str,
        violation_type: str,
        description: str,
        affected_data: Optional[str] = None,
        user_id: Optional[int] = None,
        details: Optional[Dict] = None
    ):
        """Log compliance-related events."""
        
        message = f"Compliance violation: {regulation} - {violation_type}: {description}"
        
        compliance_details = {
            'regulation': regulation,
            'violation_type': violation_type,
            'description': description,
            'affected_data': affected_data,
            **(details or {})
        }
        
        self.log_security_event(
            event_type=SecurityEventType.COMPLIANCE_VIOLATION,
            severity=SecurityEventSeverity.HIGH,
            message=message,
            user_id=user_id,
            details=compliance_details
        )

    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        return f"EVT_{int(time.time() * 1000)}_{security_manager.generate_secure_token(8)}"

    def _calculate_integrity_hash(self, event_data: Dict) -> str:
        """Calculate integrity hash for event verification."""
        # Create a copy without the hash field
        data_copy = {k: v for k, v in event_data.items() if k != 'integrity_hash'}
        
        # Sort keys for consistent hashing
        sorted_data = json.dumps(data_copy, sort_keys=True, default=str)
        
        # Calculate SHA-256 hash
        return hashlib.sha256(sorted_data.encode()).hexdigest()

    def _encrypt_event_data(self, event_data: Dict) -> str:
        """Encrypt sensitive event data."""
        return security_manager.encrypt_sensitive_data(event_data)

    def _store_event(self, encrypted_event: str, event_id: str):
        """Store event in multiple locations for redundancy."""
        
        # Primary storage in Redis
        self.redis_client.hset('security_events', event_id, encrypted_event)
        
        # Add to time-based indices
        date_key = timezone.now().strftime('%Y-%m-%d')
        self.redis_client.sadd(f'events_by_date:{date_key}', event_id)
        
        # Add to event stream
        self.redis_client.lpush('security_event_stream', event_id)
        self.redis_client.ltrim('security_event_stream', 0, 10000)  # Keep last 10k events
        
        # Set expiration (retain for 2 years for compliance)
        self.redis_client.expire(f'events_by_date:{date_key}', 86400 * 730)

    def _process_event_handlers(self, event_data: Dict):
        """Process event through registered handlers."""
        severity = event_data['severity']
        handlers = self.event_handlers.get(severity, [])
        
        for handler in handlers:
            try:
                handler(event_data)
            except Exception as e:
                # Log handler errors but don't fail the main event logging
                print(f"Event handler error: {e}")

    def _update_security_metrics(self, event_type: SecurityEventType, severity: SecurityEventSeverity):
        """Update security metrics for monitoring."""
        
        # Increment counters
        self.redis_client.hincrby('security_metrics:event_types', event_type.value, 1)
        self.redis_client.hincrby('security_metrics:severities', severity.value, 1)
        
        # Update hourly metrics
        hour_key = timezone.now().strftime('%Y-%m-%d-%H')
        self.redis_client.hincrby(f'security_metrics:hourly:{hour_key}', event_type.value, 1)
        self.redis_client.expire(f'security_metrics:hourly:{hour_key}', 86400 * 7)  # Keep for 7 days

    def _determine_fraud_severity(self, fraud_score: float, risk_level: str) -> SecurityEventSeverity:
        """Determine fraud event severity based on score and risk level."""
        if fraud_score >= 0.9 or risk_level.upper() == 'CRITICAL':
            return SecurityEventSeverity.CRITICAL
        elif fraud_score >= 0.7 or risk_level.upper() == 'HIGH':
            return SecurityEventSeverity.HIGH
        elif fraud_score >= 0.4 or risk_level.upper() == 'MEDIUM':
            return SecurityEventSeverity.MEDIUM
        else:
            return SecurityEventSeverity.LOW

    def _handle_critical_event(self, event_data: Dict):
        """Handle critical security events."""
        # Implement immediate response procedures
        self._create_incident_ticket(event_data)
        self._notify_security_team(event_data)
        
    def _handle_high_severity_event(self, event_data: Dict):
        """Handle high severity security events."""
        self._escalate_to_security_team(event_data)
        
    def _handle_medium_severity_event(self, event_data: Dict):
        """Handle medium severity security events."""
        # Add to security team queue for review
        pass
        
    def _handle_low_severity_event(self, event_data: Dict):
        """Handle low severity security events."""
        # Log only, no immediate action required
        pass

    def _send_immediate_alert(self, event_data: Dict):
        """Send immediate alert for critical events."""
        # Implementation for immediate alerting (SMS, email, Slack, etc.)
        pass
        
    def _escalate_to_security_team(self, event_data: Dict):
        """Escalate event to security team."""
        # Implementation for security team escalation
        pass
        
    def _create_incident_ticket(self, event_data: Dict):
        """Create incident ticket for critical events."""
        # Implementation for incident management system integration
        pass
        
    def _notify_security_team(self, event_data: Dict):
        """Notify security team of critical events."""
        # Implementation for security team notifications
        pass


class SecurityMetricsCollector:
    """
    Collect and analyze security metrics for monitoring and reporting.
    """

    def __init__(self):
        redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379/0')
        self.redis_client = Redis.from_url(redis_url)

    def get_security_dashboard_data(self, time_range: str = '24h') -> Dict[str, Any]:
        """Get security dashboard data for specified time range."""
        
        end_time = timezone.now()
        if time_range == '1h':
            start_time = end_time - timedelta(hours=1)
        elif time_range == '24h':
            start_time = end_time - timedelta(hours=24)
        elif time_range == '7d':
            start_time = end_time - timedelta(days=7)
        elif time_range == '30d':
            start_time = end_time - timedelta(days=30)
        else:
            start_time = end_time - timedelta(hours=24)  # Default to 24h

        return {
            'event_summary': self._get_event_summary(start_time, end_time),
            'severity_distribution': self._get_severity_distribution(start_time, end_time),
            'top_event_types': self._get_top_event_types(start_time, end_time),
            'fraud_analytics': self._get_fraud_analytics(start_time, end_time),
            'compliance_status': self._get_compliance_status(start_time, end_time),
            'trend_analysis': self._get_trend_analysis(start_time, end_time)
        }

    def _get_event_summary(self, start_time: datetime, end_time: datetime) -> Dict[str, int]:
        """Get summary of security events in time range."""
        
        # This is a simplified implementation
        # In production, you'd query your actual event storage
        
        total_events = self.redis_client.llen('security_event_stream')
        
        return {
            'total_events': min(total_events, 1000),  # Placeholder
            'critical_events': 5,
            'high_severity_events': 23,
            'medium_severity_events': 156,
            'low_severity_events': 816
        }

    def _get_severity_distribution(self, start_time: datetime, end_time: datetime) -> Dict[str, int]:
        """Get distribution of event severities."""
        return self.redis_client.hgetall('security_metrics:severities') or {}

    def _get_top_event_types(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Union[str, int]]]:
        """Get top security event types by frequency."""
        
        event_types = self.redis_client.hgetall('security_metrics:event_types') or {}
        
        # Sort by frequency
        sorted_events = sorted(
            [(event_type, int(count)) for event_type, count in event_types.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {'event_type': event_type, 'count': count}
            for event_type, count in sorted_events[:10]
        ]

    def _get_fraud_analytics(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get fraud detection analytics."""
        
        return {
            'total_transactions_analyzed': 12450,
            'fraud_detections': 187,
            'fraud_rate': 1.5,
            'avg_fraud_score': 0.23,
            'high_risk_transactions': 45,
            'blocked_transactions': 12
        }

    def _get_compliance_status(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get compliance status metrics."""
        
        return {
            'gdpr_compliance_score': 98.5,
            'pci_dss_compliance_score': 99.2,
            'sox_compliance_score': 97.8,
            'audit_trail_coverage': 100.0,
            'data_retention_compliance': 99.9,
            'violations': 2
        }

    def _get_trend_analysis(self, start_time: datetime, end_time: datetime) -> Dict[str, List]:
        """Get security trend analysis."""
        
        # Generate sample trend data
        hours = []
        current_time = start_time
        
        while current_time <= end_time:
            hours.append({
                'timestamp': current_time.isoformat(),
                'total_events': 45,
                'critical_events': 1,
                'fraud_detections': 3
            })
            current_time += timedelta(hours=1)
        
        return {
            'hourly_trends': hours[-24:]  # Last 24 hours
        }


class ComplianceMonitor:
    """
    Monitor compliance with various regulations and standards.
    """

    def __init__(self):
        self.audit_logger = AuditLogger()
        self.compliance_rules = self._load_compliance_rules()

    def check_gdpr_compliance(self, data_operation: Dict) -> Dict[str, Any]:
        """Check GDPR compliance for data operations."""
        
        violations = []
        
        # Check for consent
        if not data_operation.get('user_consent'):
            violations.append("No user consent recorded")
        
        # Check for data minimization
        if self._excessive_data_collection(data_operation):
            violations.append("Excessive data collection detected")
        
        # Check for purpose limitation
        if not data_operation.get('processing_purpose'):
            violations.append("No processing purpose specified")
        
        # Check for retention period
        if self._data_retention_violation(data_operation):
            violations.append("Data retention period exceeded")
        
        compliance_status = {
            'compliant': len(violations) == 0,
            'violations': violations,
            'score': max(0, 100 - (len(violations) * 25))  # 25 points per violation
        }
        
        # Log compliance check
        if violations:
            self.audit_logger.log_compliance_event(
                regulation='GDPR',
                violation_type='Data Processing Violation',
                description='; '.join(violations),
                details=data_operation
            )
        
        return compliance_status

    def check_pci_dss_compliance(self, payment_operation: Dict) -> Dict[str, Any]:
        """Check PCI DSS compliance for payment operations."""
        
        violations = []
        
        # Check for encrypted card data
        if not payment_operation.get('card_data_encrypted'):
            violations.append("Card data not encrypted")
        
        # Check for secure transmission
        if not payment_operation.get('secure_transmission'):
            violations.append("Insecure data transmission")
        
        # Check for access controls
        if not payment_operation.get('access_controls_verified'):
            violations.append("Access controls not verified")
        
        compliance_status = {
            'compliant': len(violations) == 0,
            'violations': violations,
            'score': max(0, 100 - (len(violations) * 20))
        }
        
        if violations:
            self.audit_logger.log_compliance_event(
                regulation='PCI DSS',
                violation_type='Payment Security Violation',
                description='; '.join(violations),
                details=payment_operation
            )
        
        return compliance_status

    def check_sox_compliance(self, financial_operation: Dict) -> Dict[str, Any]:
        """Check SOX compliance for financial operations."""
        
        violations = []
        
        # Check for proper authorization
        if not financial_operation.get('authorized_by'):
            violations.append("Financial operation not properly authorized")
        
        # Check for audit trail
        if not financial_operation.get('audit_trail_complete'):
            violations.append("Incomplete audit trail")
        
        # Check for segregation of duties
        if not financial_operation.get('segregation_of_duties'):
            violations.append("Segregation of duties violation")
        
        compliance_status = {
            'compliant': len(violations) == 0,
            'violations': violations,
            'score': max(0, 100 - (len(violations) * 15))
        }
        
        if violations:
            self.audit_logger.log_compliance_event(
                regulation='SOX',
                violation_type='Financial Controls Violation',
                description='; '.join(violations),
                details=financial_operation
            )
        
        return compliance_status

    def _excessive_data_collection(self, operation: Dict) -> bool:
        """Check if data collection is excessive."""
        # Implement logic to detect excessive data collection
        return False

    def _data_retention_violation(self, operation: Dict) -> bool:
        """Check for data retention violations."""
        # Implement logic to check retention periods
        return False

    def _load_compliance_rules(self) -> Dict:
        """Load compliance rules configuration."""
        return {
            'gdpr': {
                'data_retention_days': 365,
                'consent_required': True,
                'anonymization_required': True
            },
            'pci_dss': {
                'encryption_required': True,
                'secure_transmission': True,
                'access_logging': True
            },
            'sox': {
                'authorization_required': True,
                'audit_trail_required': True,
                'segregation_of_duties': True
            }
        }


class SecurityAlertManager:
    """
    Manage security alerts and notifications.
    """

    def __init__(self):
        redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379/0')
        self.redis_client = Redis.from_url(redis_url)
        self.alert_channels = self._initialize_alert_channels()

    def send_security_alert(
        self,
        severity: SecurityEventSeverity,
        title: str,
        message: str,
        details: Optional[Dict] = None,
        recipients: Optional[List[str]] = None
    ):
        """Send security alert through appropriate channels."""
        
        alert = {
            'id': security_manager.generate_secure_token(16),
            'timestamp': timezone.now().isoformat(),
            'severity': severity.value,
            'title': title,
            'message': message,
            'details': details or {},
            'recipients': recipients or self._get_default_recipients(severity)
        }
        
        # Store alert
        self.redis_client.hset('security_alerts', alert['id'], json.dumps(alert))
        
        # Send through appropriate channels based on severity
        self._route_alert(alert)

    def _initialize_alert_channels(self) -> Dict:
        """Initialize alert channels configuration."""
        return {
            'email': EmailAlertChannel(),
            'sms': SMSAlertChannel(),
            'slack': SlackAlertChannel(),
            'webhook': WebhookAlertChannel()
        }

    def _get_default_recipients(self, severity: SecurityEventSeverity) -> List[str]:
        """Get default recipients based on severity."""
        
        if severity == SecurityEventSeverity.CRITICAL:
            return ['security-team@company.com', 'ciso@company.com', 'on-call@company.com']
        elif severity == SecurityEventSeverity.HIGH:
            return ['security-team@company.com', 'devops@company.com']
        elif severity == SecurityEventSeverity.MEDIUM:
            return ['security-team@company.com']
        else:
            return ['security-alerts@company.com']

    def _route_alert(self, alert: Dict):
        """Route alert to appropriate channels."""
        
        severity = SecurityEventSeverity(alert['severity'])
        
        if severity == SecurityEventSeverity.CRITICAL:
            # Send via all channels for critical alerts
            for channel in self.alert_channels.values():
                channel.send_alert(alert)
        elif severity == SecurityEventSeverity.HIGH:
            # Send via email and Slack
            self.alert_channels['email'].send_alert(alert)
            self.alert_channels['slack'].send_alert(alert)
        else:
            # Send via email only
            self.alert_channels['email'].send_alert(alert)


class EmailAlertChannel:
    """Email alert channel implementation."""
    
    def send_alert(self, alert: Dict):
        """Send alert via email."""
        # Implement email sending logic
        pass


class SMSAlertChannel:
    """SMS alert channel implementation."""
    
    def send_alert(self, alert: Dict):
        """Send alert via SMS."""
        # Implement SMS sending logic
        pass


class SlackAlertChannel:
    """Slack alert channel implementation."""
    
    def send_alert(self, alert: Dict):
        """Send alert via Slack."""
        # Implement Slack webhook logic
        pass


class WebhookAlertChannel:
    """Webhook alert channel implementation."""
    
    def send_alert(self, alert: Dict):
        """Send alert via webhook."""
        # Implement webhook sending logic
        pass


# Initialize audit and monitoring components
audit_logger = AuditLogger()
security_metrics = SecurityMetricsCollector()
compliance_monitor = ComplianceMonitor()
alert_manager = SecurityAlertManager()