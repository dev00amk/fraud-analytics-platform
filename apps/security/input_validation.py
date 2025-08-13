"""
Comprehensive Input Validation and Sanitization Framework
Enterprise-grade protection against injection attacks and malicious input
"""

import base64
import html
import re
import urllib.parse
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import bleach
from django.core.exceptions import ValidationError
from django.core.validators import validate_email, validate_ipv4_address, validate_ipv6_address
from django.utils import timezone

from .audit_monitoring import audit_logger, SecurityEventType, SecurityEventSeverity


class InputValidator:
    """
    Comprehensive input validation with security-focused sanitization.
    Protects against OWASP Top 10 input-based vulnerabilities.
    """

    def __init__(self):
        self.sql_injection_patterns = self._load_sql_patterns()
        self.xss_patterns = self._load_xss_patterns()
        self.command_injection_patterns = self._load_command_patterns()
        self.path_traversal_patterns = self._load_path_patterns()
        
        # Allowed HTML tags and attributes for rich text content
        self.allowed_tags = ['p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li', 'a', 'span']
        self.allowed_attributes = {
            'a': ['href', 'title'],
            'span': ['class'],
        }

    def validate_and_sanitize(
        self, 
        input_data: Dict[str, Any], 
        validation_schema: Dict[str, Dict],
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate and sanitize input data based on schema.
        
        Args:
            input_data: Raw input data
            validation_schema: Validation rules for each field
            context: Context for logging (e.g., 'transaction_creation')
        
        Returns:
            Sanitized and validated data
        
        Raises:
            ValidationError: If validation fails
        """
        
        sanitized_data = {}
        validation_errors = []
        
        for field_name, field_value in input_data.items():
            try:
                # Get validation rules for this field
                field_rules = validation_schema.get(field_name, {})
                
                # Skip validation if field not in schema and not required
                if not field_rules and field_name not in validation_schema:
                    continue
                
                # Apply validation and sanitization
                sanitized_value = self._validate_field(
                    field_name, field_value, field_rules, context
                )
                
                sanitized_data[field_name] = sanitized_value
                
            except ValidationError as e:
                validation_errors.append(f"{field_name}: {str(e)}")
                
                # Log validation failure
                audit_logger.log_security_event(
                    event_type=SecurityEventType.SECURITY_VIOLATION,
                    severity=SecurityEventSeverity.MEDIUM,
                    message=f"Input validation failed for field {field_name}",
                    details={
                        'field': field_name,
                        'value': str(field_value)[:100],  # Truncated for security
                        'error': str(e),
                        'context': context
                    }
                )
        
        # Check for required fields
        for field_name, field_rules in validation_schema.items():
            if field_rules.get('required', False) and field_name not in input_data:
                validation_errors.append(f"{field_name}: This field is required")
        
        if validation_errors:
            raise ValidationError(validation_errors)
        
        return sanitized_data

    def _validate_field(
        self, 
        field_name: str, 
        field_value: Any, 
        field_rules: Dict, 
        context: Optional[str]
    ) -> Any:
        """Validate individual field based on rules."""
        
        if field_value is None:
            if field_rules.get('required', False):
                raise ValidationError("This field is required")
            return None
        
        # Get field type and apply appropriate validation
        field_type = field_rules.get('type', 'string')
        
        if field_type == 'string':
            return self._validate_string(field_name, field_value, field_rules, context)
        elif field_type == 'integer':
            return self._validate_integer(field_name, field_value, field_rules)
        elif field_type == 'float':
            return self._validate_float(field_name, field_value, field_rules)
        elif field_type == 'decimal':
            return self._validate_decimal(field_name, field_value, field_rules)
        elif field_type == 'email':
            return self._validate_email(field_name, field_value, field_rules)
        elif field_type == 'url':
            return self._validate_url(field_name, field_value, field_rules)
        elif field_type == 'ip_address':
            return self._validate_ip_address(field_name, field_value, field_rules)
        elif field_type == 'datetime':
            return self._validate_datetime(field_name, field_value, field_rules)
        elif field_type == 'json':
            return self._validate_json(field_name, field_value, field_rules)
        elif field_type == 'list':
            return self._validate_list(field_name, field_value, field_rules, context)
        elif field_type == 'rich_text':
            return self._validate_rich_text(field_name, field_value, field_rules)
        else:
            # Default to string validation
            return self._validate_string(field_name, field_value, field_rules, context)

    def _validate_string(
        self, 
        field_name: str, 
        value: Any, 
        rules: Dict, 
        context: Optional[str]
    ) -> str:
        """Validate and sanitize string input."""
        
        # Convert to string
        str_value = str(value) if value is not None else ''
        
        # Length validation
        min_length = rules.get('min_length', 0)
        max_length = rules.get('max_length', 10000)
        
        if len(str_value) < min_length:
            raise ValidationError(f"Must be at least {min_length} characters long")
        
        if len(str_value) > max_length:
            raise ValidationError(f"Must be no more than {max_length} characters long")
        
        # Security validation
        self._check_security_threats(field_name, str_value, context)
        
        # Pattern validation
        pattern = rules.get('pattern')
        if pattern and not re.match(pattern, str_value):
            raise ValidationError("Invalid format")
        
        # Sanitization
        sanitized_value = self._sanitize_string(str_value, rules)
        
        return sanitized_value

    def _validate_integer(self, field_name: str, value: Any, rules: Dict) -> int:
        """Validate integer input."""
        
        try:
            int_value = int(value)
        except (ValueError, TypeError):
            raise ValidationError("Must be a valid integer")
        
        # Range validation
        min_value = rules.get('min_value')
        max_value = rules.get('max_value')
        
        if min_value is not None and int_value < min_value:
            raise ValidationError(f"Must be at least {min_value}")
        
        if max_value is not None and int_value > max_value:
            raise ValidationError(f"Must be no more than {max_value}")
        
        return int_value

    def _validate_float(self, field_name: str, value: Any, rules: Dict) -> float:
        """Validate float input."""
        
        try:
            float_value = float(value)
        except (ValueError, TypeError):
            raise ValidationError("Must be a valid number")
        
        # Range validation
        min_value = rules.get('min_value')
        max_value = rules.get('max_value')
        
        if min_value is not None and float_value < min_value:
            raise ValidationError(f"Must be at least {min_value}")
        
        if max_value is not None and float_value > max_value:
            raise ValidationError(f"Must be no more than {max_value}")
        
        return float_value

    def _validate_decimal(self, field_name: str, value: Any, rules: Dict) -> Decimal:
        """Validate decimal input for financial data."""
        
        try:
            decimal_value = Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            raise ValidationError("Must be a valid decimal number")
        
        # Precision validation
        max_digits = rules.get('max_digits', 10)
        decimal_places = rules.get('decimal_places', 2)
        
        # Check total digits
        sign, digits, exponent = decimal_value.as_tuple()
        if len(digits) > max_digits:
            raise ValidationError(f"Must have no more than {max_digits} digits")
        
        # Check decimal places
        if exponent < -decimal_places:
            raise ValidationError(f"Must have no more than {decimal_places} decimal places")
        
        # Range validation
        min_value = rules.get('min_value')
        max_value = rules.get('max_value')
        
        if min_value is not None and decimal_value < Decimal(str(min_value)):
            raise ValidationError(f"Must be at least {min_value}")
        
        if max_value is not None and decimal_value > Decimal(str(max_value)):
            raise ValidationError(f"Must be no more than {max_value}")
        
        return decimal_value

    def _validate_email(self, field_name: str, value: Any, rules: Dict) -> str:
        """Validate email address."""
        
        email_str = str(value).strip().lower()
        
        try:
            validate_email(email_str)
        except ValidationError:
            raise ValidationError("Invalid email address")
        
        # Additional security checks
        self._check_security_threats(field_name, email_str, 'email_validation')
        
        return email_str

    def _validate_url(self, field_name: str, value: Any, rules: Dict) -> str:
        """Validate URL."""
        
        url_str = str(value).strip()
        
        # Basic URL format validation
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        if not url_pattern.match(url_str):
            raise ValidationError("Invalid URL format")
        
        # Security checks
        parsed_url = urllib.parse.urlparse(url_str)
        
        # Check for dangerous schemes
        if parsed_url.scheme not in ['http', 'https']:
            raise ValidationError("Only HTTP and HTTPS URLs are allowed")
        
        # Check for dangerous hosts (implement your own blacklist)
        dangerous_hosts = ['localhost', '127.0.0.1', '0.0.0.0']
        if rules.get('allow_local', False) is False and parsed_url.hostname in dangerous_hosts:
            raise ValidationError("Local URLs are not allowed")
        
        return url_str

    def _validate_ip_address(self, field_name: str, value: Any, rules: Dict) -> str:
        """Validate IP address."""
        
        ip_str = str(value).strip()
        
        try:
            validate_ipv4_address(ip_str)
            return ip_str
        except ValidationError:
            try:
                validate_ipv6_address(ip_str)
                return ip_str
            except ValidationError:
                raise ValidationError("Invalid IP address")

    def _validate_datetime(self, field_name: str, value: Any, rules: Dict) -> datetime:
        """Validate datetime input."""
        
        if isinstance(value, datetime):
            return value
        
        # Try to parse string datetime
        datetime_str = str(value).strip()
        
        # Common datetime formats to try
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%Y-%m-%d'
        ]
        
        for fmt in formats:
            try:
                parsed_datetime = datetime.strptime(datetime_str, fmt)
                
                # Make timezone aware if required
                if rules.get('timezone_aware', True) and timezone.is_naive(parsed_datetime):
                    parsed_datetime = timezone.make_aware(parsed_datetime)
                
                return parsed_datetime
            except ValueError:
                continue
        
        raise ValidationError("Invalid datetime format")

    def _validate_json(self, field_name: str, value: Any, rules: Dict) -> Union[Dict, List]:
        """Validate JSON input."""
        
        if isinstance(value, (dict, list)):
            json_value = value
        else:
            try:
                import json
                json_value = json.loads(str(value))
            except (json.JSONDecodeError, ValueError):
                raise ValidationError("Invalid JSON format")
        
        # Check JSON size
        max_size = rules.get('max_size', 100000)  # 100KB default
        json_str = str(json_value)
        if len(json_str) > max_size:
            raise ValidationError(f"JSON too large (max {max_size} characters)")
        
        # Recursively validate JSON content for security threats
        self._validate_json_security(json_value, field_name)
        
        return json_value

    def _validate_list(
        self, 
        field_name: str, 
        value: Any, 
        rules: Dict, 
        context: Optional[str]
    ) -> List:
        """Validate list input."""
        
        if not isinstance(value, list):
            raise ValidationError("Must be a list")
        
        # Length validation
        min_items = rules.get('min_items', 0)
        max_items = rules.get('max_items', 1000)
        
        if len(value) < min_items:
            raise ValidationError(f"Must contain at least {min_items} items")
        
        if len(value) > max_items:
            raise ValidationError(f"Must contain no more than {max_items} items")
        
        # Validate individual items
        item_rules = rules.get('item_rules', {})
        validated_items = []
        
        for i, item in enumerate(value):
            try:
                validated_item = self._validate_field(
                    f"{field_name}[{i}]", item, item_rules, context
                )
                validated_items.append(validated_item)
            except ValidationError as e:
                raise ValidationError(f"Item {i}: {str(e)}")
        
        return validated_items

    def _validate_rich_text(self, field_name: str, value: Any, rules: Dict) -> str:
        """Validate and sanitize rich text content."""
        
        html_content = str(value)
        
        # Use bleach to sanitize HTML
        sanitized_html = bleach.clean(
            html_content,
            tags=self.allowed_tags,
            attributes=self.allowed_attributes,
            strip=True
        )
        
        # Additional XSS protection
        self._check_xss_patterns(html_content, field_name)
        
        return sanitized_html

    def _sanitize_string(self, value: str, rules: Dict) -> str:
        """Apply string sanitization based on rules."""
        
        sanitized = value
        
        # HTML escape if requested
        if rules.get('escape_html', True):
            sanitized = html.escape(sanitized)
        
        # SQL escape if requested
        if rules.get('escape_sql', True):
            sanitized = self._escape_sql_chars(sanitized)
        
        # Trim whitespace if requested
        if rules.get('trim', True):
            sanitized = sanitized.strip()
        
        # Convert to lowercase if requested
        if rules.get('lowercase', False):
            sanitized = sanitized.lower()
        
        # Convert to uppercase if requested
        if rules.get('uppercase', False):
            sanitized = sanitized.upper()
        
        return sanitized

    def _check_security_threats(self, field_name: str, value: str, context: Optional[str]):
        """Check for various security threats in input."""
        
        # Check for SQL injection
        if self._contains_sql_injection(value):
            audit_logger.log_security_event(
                event_type=SecurityEventType.SECURITY_VIOLATION,
                severity=SecurityEventSeverity.HIGH,
                message=f"SQL injection attempt detected in field {field_name}",
                details={'field': field_name, 'context': context}
            )
            raise ValidationError("Input contains potentially malicious content")
        
        # Check for XSS
        if self._contains_xss(value):
            audit_logger.log_security_event(
                event_type=SecurityEventType.SECURITY_VIOLATION,
                severity=SecurityEventSeverity.HIGH,
                message=f"XSS attempt detected in field {field_name}",
                details={'field': field_name, 'context': context}
            )
            raise ValidationError("Input contains potentially malicious content")
        
        # Check for command injection
        if self._contains_command_injection(value):
            audit_logger.log_security_event(
                event_type=SecurityEventType.SECURITY_VIOLATION,
                severity=SecurityEventSeverity.HIGH,
                message=f"Command injection attempt detected in field {field_name}",
                details={'field': field_name, 'context': context}
            )
            raise ValidationError("Input contains potentially malicious content")
        
        # Check for path traversal
        if self._contains_path_traversal(value):
            audit_logger.log_security_event(
                event_type=SecurityEventType.SECURITY_VIOLATION,
                severity=SecurityEventSeverity.MEDIUM,
                message=f"Path traversal attempt detected in field {field_name}",
                details={'field': field_name, 'context': context}
            )
            raise ValidationError("Input contains potentially malicious content")

    def _contains_sql_injection(self, value: str) -> bool:
        """Check for SQL injection patterns."""
        value_lower = value.lower()
        
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, value_lower):
                return True
        
        return False

    def _contains_xss(self, value: str) -> bool:
        """Check for XSS patterns."""
        return self._check_xss_patterns(value)

    def _check_xss_patterns(self, value: str, field_name: Optional[str] = None) -> bool:
        """Check for XSS patterns in content."""
        value_lower = value.lower()
        
        for pattern in self.xss_patterns:
            if re.search(pattern, value_lower):
                return True
        
        return False

    def _contains_command_injection(self, value: str) -> bool:
        """Check for command injection patterns."""
        
        for pattern in self.command_injection_patterns:
            if re.search(pattern, value):
                return True
        
        return False

    def _contains_path_traversal(self, value: str) -> bool:
        """Check for path traversal patterns."""
        
        for pattern in self.path_traversal_patterns:
            if re.search(pattern, value):
                return True
        
        return False

    def _validate_json_security(self, json_value: Union[Dict, List], field_name: str):
        """Recursively validate JSON for security threats."""
        
        if isinstance(json_value, dict):
            for key, value in json_value.items():
                # Check key for threats
                self._check_security_threats(f"{field_name}.{key}", str(key), 'json_key')
                
                # Recursively check value
                if isinstance(value, (dict, list)):
                    self._validate_json_security(value, f"{field_name}.{key}")
                else:
                    self._check_security_threats(f"{field_name}.{key}", str(value), 'json_value')
        
        elif isinstance(json_value, list):
            for i, item in enumerate(json_value):
                if isinstance(item, (dict, list)):
                    self._validate_json_security(item, f"{field_name}[{i}]")
                else:
                    self._check_security_threats(f"{field_name}[{i}]", str(item), 'json_array_item')

    def _escape_sql_chars(self, value: str) -> str:
        """Escape SQL special characters."""
        
        # Basic SQL character escaping
        sql_chars = {
            "'": "''",
            '"': '""',
            '\\': '\\\\',
            '\n': '\\n',
            '\r': '\\r',
            '\t': '\\t'
        }
        
        for char, escaped in sql_chars.items():
            value = value.replace(char, escaped)
        
        return value

    def _load_sql_patterns(self) -> List[str]:
        """Load SQL injection detection patterns."""
        return [
            r"\b(select|insert|update|delete|drop|create|alter|exec|execute)\b.*\b(from|into|set|where|values)\b",
            r"\b(union|or|and)\b.*\b(select|insert)\b",
            r"('|\").*(or|and).*('|\")",
            r"\b(exec|execute)\b.*\b(sp_|xp_)\b",
            r";\s*(select|insert|update|delete|drop)",
            r"--.*$",
            r"/\*.*\*/",
            r"\bcast\s*\(",
            r"\bconvert\s*\(",
            r"\bchar\s*\(",
            r"\bhex\s*\(",
            r"\bload_file\s*\(",
            r"\boutfile\s*\(",
        ]

    def _load_xss_patterns(self) -> List[str]:
        """Load XSS detection patterns."""
        return [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"vbscript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>.*?</iframe>",
            r"<object[^>]*>.*?</object>",
            r"<embed[^>]*>",
            r"<link[^>]*>",
            r"<meta[^>]*>",
            r"<style[^>]*>.*?</style>",
            r"expression\s*\(",
            r"url\s*\(",
            r"@import",
            r"<img[^>]*src\s*=\s*[\"']?javascript:",
            r"<svg[^>]*onload\s*=",
        ]

    def _load_command_patterns(self) -> List[str]:
        """Load command injection detection patterns."""
        return [
            r"[;&|`]",
            r"\$\(",
            r"`.*`",
            r"\|\s*(cat|ls|pwd|whoami|id|uname)",
            r"(curl|wget|nc|telnet)\s+",
            r"(rm|mv|cp|chmod|chown)\s+",
            r">\s*/",
            r"<\s*/",
            r"\|\s*sh",
            r"\|\s*bash",
            r"&&\s*(cat|ls|pwd)",
            r"\|\|\s*(cat|ls|pwd)",
        ]

    def _load_path_patterns(self) -> List[str]:
        """Load path traversal detection patterns."""
        return [
            r"\.\./",
            r"\.\.\\",
            r"~/",
            r"%2e%2e%2f",
            r"%2e%2e\\",
            r"%252e%252e%252f",
            r"\.\.%2f",
            r"\.\.%5c",
            r"%c0%af",
            r"%c1%9c",
        ]


class TransactionValidator:
    """
    Specialized validator for financial transaction data.
    Implements additional security measures for financial operations.
    """

    def __init__(self):
        self.base_validator = InputValidator()

    def validate_transaction(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate transaction data with financial-specific rules."""
        
        schema = {
            'transaction_id': {
                'type': 'string',
                'required': True,
                'min_length': 8,
                'max_length': 64,
                'pattern': r'^[A-Za-z0-9_-]+$'  # Alphanumeric, underscore, hyphen only
            },
            'user_id': {
                'type': 'string',
                'required': True,
                'min_length': 1,
                'max_length': 64,
                'pattern': r'^[A-Za-z0-9_-]+$'
            },
            'amount': {
                'type': 'decimal',
                'required': True,
                'min_value': 0.01,
                'max_value': 1000000.00,
                'max_digits': 10,
                'decimal_places': 2
            },
            'currency': {
                'type': 'string',
                'required': True,
                'pattern': r'^[A-Z]{3}$'  # ISO 4217 currency codes
            },
            'merchant_id': {
                'type': 'string',
                'required': True,
                'min_length': 1,
                'max_length': 64,
                'pattern': r'^[A-Za-z0-9_-]+$'
            },
            'payment_method': {
                'type': 'string',
                'required': True,
                'pattern': r'^(credit_card|debit_card|bank_transfer|digital_wallet)$'
            },
            'timestamp': {
                'type': 'datetime',
                'required': True,
                'timezone_aware': True
            },
            'ip_address': {
                'type': 'ip_address',
                'required': True
            },
            'user_agent': {
                'type': 'string',
                'required': False,
                'max_length': 500,
                'escape_html': True
            },
            'device_fingerprint': {
                'type': 'string',
                'required': False,
                'max_length': 256,
                'pattern': r'^[A-Za-z0-9+/=]*$'  # Base64 pattern
            },
            'metadata': {
                'type': 'json',
                'required': False,
                'max_size': 10000  # 10KB max
            }
        }
        
        return self.base_validator.validate_and_sanitize(
            transaction_data, schema, 'transaction_validation'
        )


class APIRequestValidator:
    """
    Specialized validator for API requests.
    Provides endpoint-specific validation rules.
    """

    def __init__(self):
        self.base_validator = InputValidator()
        self.endpoint_schemas = self._load_endpoint_schemas()

    def validate_api_request(
        self, 
        endpoint: str, 
        method: str, 
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate API request based on endpoint and method."""
        
        schema_key = f"{method.upper()}:{endpoint}"
        schema = self.endpoint_schemas.get(schema_key, {})
        
        if not schema:
            # Use generic validation if no specific schema
            schema = self._get_generic_api_schema()
        
        return self.base_validator.validate_and_sanitize(
            request_data, schema, f'api_request:{endpoint}'
        )

    def _load_endpoint_schemas(self) -> Dict[str, Dict]:
        """Load validation schemas for API endpoints."""
        return {
            'POST:/api/v1/analyze': {
                'transaction_id': {'type': 'string', 'required': True, 'max_length': 64},
                'user_id': {'type': 'string', 'required': True, 'max_length': 64},
                'amount': {'type': 'decimal', 'required': True, 'min_value': 0.01},
                'currency': {'type': 'string', 'required': True, 'pattern': r'^[A-Z]{3}$'},
                'timestamp': {'type': 'datetime', 'required': True},
                'metadata': {'type': 'json', 'required': False, 'max_size': 5000}
            },
            'POST:/api/v1/transactions': {
                'page': {'type': 'integer', 'min_value': 1, 'max_value': 1000},
                'limit': {'type': 'integer', 'min_value': 1, 'max_value': 100},
                'status': {'type': 'string', 'pattern': r'^(pending|approved|declined|flagged)$'},
                'date_from': {'type': 'datetime', 'required': False},
                'date_to': {'type': 'datetime', 'required': False}
            },
            'POST:/api/v1/cases': {
                'title': {'type': 'string', 'required': True, 'max_length': 200},
                'description': {'type': 'rich_text', 'required': True, 'max_length': 5000},
                'priority': {'type': 'string', 'pattern': r'^(low|medium|high|critical)$'},
                'assigned_to': {'type': 'string', 'required': False, 'max_length': 64}
            }
        }

    def _get_generic_api_schema(self) -> Dict[str, Dict]:
        """Get generic API validation schema."""
        return {
            'page': {'type': 'integer', 'min_value': 1, 'max_value': 1000},
            'limit': {'type': 'integer', 'min_value': 1, 'max_value': 100},
            'sort': {'type': 'string', 'max_length': 50, 'pattern': r'^[a-zA-Z_]+$'},
            'order': {'type': 'string', 'pattern': r'^(asc|desc)$'}
        }


# Initialize validators
input_validator = InputValidator()
transaction_validator = TransactionValidator()
api_request_validator = APIRequestValidator()