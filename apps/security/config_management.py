"""
Secure Configuration Management System
Enterprise-grade secure configuration with secrets management and environment-aware settings
"""

import os
import json
import hashlib
import hmac
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from cryptography.fernet import Fernet
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

from .security_manager import security_manager
from .audit_monitoring import audit_logger, SecurityEventType, SecurityEventSeverity


@dataclass
class ConfigValue:
    """Represents a configuration value with metadata."""
    value: Any
    is_encrypted: bool = False
    is_sensitive: bool = False
    environment: str = 'all'
    description: str = ''
    last_modified: str = ''
    version: str = '1.0'


class SecureConfigManager:
    """
    Secure configuration management with:
    - Environment-specific configurations
    - Secrets encryption and rotation
    - Configuration validation
    - Audit logging
    - Hot reload capabilities
    """

    def __init__(self, config_dir: str = None):
        self.config_dir = Path(config_dir or settings.BASE_DIR / 'config')
        self.environment = os.getenv('DJANGO_ENV', 'development')
        self.configs = {}
        self.encryption_key = self._get_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # Load configurations
        self._load_configurations()
        
        # Configuration change watchers
        self.change_handlers = []

    def _get_encryption_key(self) -> bytes:
        """Get or generate encryption key for configuration encryption."""
        key_env = os.getenv('CONFIG_ENCRYPTION_KEY')
        if key_env:
            return key_env.encode()
        
        # Generate key if not exists (for development)
        key_file = self.config_dir / '.config_key'
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            self.config_dir.mkdir(exist_ok=True)
            with open(key_file, 'wb') as f:
                f.write(key)
            
            # Set restrictive permissions
            os.chmod(key_file, 0o600)
            return key

    def _load_configurations(self):
        """Load configurations from files and environment variables."""
        
        # Load base configuration
        self._load_config_file('base.json')
        
        # Load environment-specific configuration
        env_config_file = f'{self.environment}.json'
        self._load_config_file(env_config_file)
        
        # Load encrypted secrets
        secrets_file = f'secrets.{self.environment}.encrypted'
        self._load_encrypted_config(secrets_file)
        
        # Override with environment variables
        self._load_environment_variables()
        
        # Validate configuration
        self._validate_configuration()

    def _load_config_file(self, filename: str):
        """Load configuration from JSON file."""
        config_file = self.config_dir / filename
        
        if not config_file.exists():
            return
        
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)
            
            for key, value in data.items():
                if isinstance(value, dict) and 'value' in value:
                    # Enhanced config format with metadata
                    config_value = ConfigValue(
                        value=value['value'],
                        is_encrypted=value.get('encrypted', False),
                        is_sensitive=value.get('sensitive', False),
                        environment=value.get('environment', 'all'),
                        description=value.get('description', ''),
                        version=value.get('version', '1.0')
                    )
                else:
                    # Simple value
                    config_value = ConfigValue(value=value)
                
                # Only load if appropriate for current environment
                if (config_value.environment == 'all' or 
                    config_value.environment == self.environment):
                    self.configs[key] = config_value
                    
        except (json.JSONDecodeError, FileNotFoundError) as e:
            audit_logger.log_security_event(
                event_type=SecurityEventType.SYSTEM_CONFIGURATION_CHANGE,
                severity=SecurityEventSeverity.MEDIUM,
                message=f"Failed to load configuration file {filename}",
                details={'error': str(e), 'file': filename}
            )

    def _load_encrypted_config(self, filename: str):
        """Load encrypted configuration file."""
        config_file = self.config_dir / filename
        
        if not config_file.exists():
            return
        
        try:
            with open(config_file, 'rb') as f:
                encrypted_data = f.read()
            
            # Decrypt configuration
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            config_data = json.loads(decrypted_data.decode())
            
            for key, value in config_data.items():
                config_value = ConfigValue(
                    value=value,
                    is_encrypted=True,
                    is_sensitive=True,
                    environment=self.environment
                )
                self.configs[key] = config_value
                
        except Exception as e:
            audit_logger.log_security_event(
                event_type=SecurityEventType.SYSTEM_CONFIGURATION_CHANGE,
                severity=SecurityEventSeverity.HIGH,
                message=f"Failed to decrypt configuration file {filename}",
                details={'error': str(e), 'file': filename}
            )

    def _load_environment_variables(self):
        """Load configuration from environment variables."""
        
        # Define environment variable prefixes for different config types
        prefixes = {
            'FRAUD_': False,  # Regular config
            'FRAUD_SECRET_': True,  # Sensitive config
        }
        
        for prefix, is_sensitive in prefixes.items():
            for env_key, env_value in os.environ.items():
                if env_key.startswith(prefix):
                    # Convert environment variable name to config key
                    config_key = env_key[len(prefix):].lower().replace('_', '.')
                    
                    # Parse value
                    parsed_value = self._parse_env_value(env_value)
                    
                    config_value = ConfigValue(
                        value=parsed_value,
                        is_sensitive=is_sensitive,
                        environment=self.environment
                    )
                    
                    self.configs[config_key] = config_value

    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        
        # Try to parse as JSON first (for complex types)
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        # Parse common boolean values
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        elif value.lower() in ('false', 'no', '0', 'off'):
            return False
        
        # Try to parse as integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try to parse as float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value

    def _validate_configuration(self):
        """Validate loaded configuration against schema."""
        
        required_configs = self._get_required_configs()
        missing_configs = []
        
        for config_key, config_info in required_configs.items():
            if config_key not in self.configs:
                missing_configs.append(config_key)
            else:
                # Validate type and constraints
                config_value = self.configs[config_key]
                if not self._validate_config_value(config_value, config_info):
                    audit_logger.log_security_event(
                        event_type=SecurityEventType.SYSTEM_CONFIGURATION_CHANGE,
                        severity=SecurityEventSeverity.MEDIUM,
                        message=f"Configuration validation failed for {config_key}",
                        details={'config_key': config_key, 'environment': self.environment}
                    )
        
        if missing_configs:
            error_msg = f"Missing required configurations: {', '.join(missing_configs)}"
            audit_logger.log_security_event(
                event_type=SecurityEventType.SYSTEM_CONFIGURATION_CHANGE,
                severity=SecurityEventSeverity.HIGH,
                message=error_msg,
                details={'missing_configs': missing_configs, 'environment': self.environment}
            )
            raise ImproperlyConfigured(error_msg)

    def _get_required_configs(self) -> Dict[str, Dict]:
        """Get required configuration schema."""
        return {
            'database.url': {
                'type': str,
                'required': True,
                'sensitive': True,
                'description': 'Database connection URL'
            },
            'redis.url': {
                'type': str,
                'required': True,
                'sensitive': True,
                'description': 'Redis connection URL'
            },
            'security.secret_key': {
                'type': str,
                'required': True,
                'sensitive': True,
                'min_length': 50,
                'description': 'Django secret key'
            },
            'security.encryption_key': {
                'type': str,
                'required': True,
                'sensitive': True,
                'description': 'Application encryption key'
            },
            'fraud.detection.enabled': {
                'type': bool,
                'required': True,
                'default': True,
                'description': 'Enable fraud detection'
            },
            'api.rate_limit.requests_per_minute': {
                'type': int,
                'required': False,
                'default': 100,
                'min_value': 1,
                'max_value': 10000,
                'description': 'API rate limit'
            }
        }

    def _validate_config_value(self, config_value: ConfigValue, config_info: Dict) -> bool:
        """Validate individual configuration value."""
        
        value = config_value.value
        
        # Type validation
        expected_type = config_info.get('type')
        if expected_type and not isinstance(value, expected_type):
            return False
        
        # String length validation
        if isinstance(value, str):
            min_length = config_info.get('min_length')
            max_length = config_info.get('max_length')
            
            if min_length and len(value) < min_length:
                return False
            if max_length and len(value) > max_length:
                return False
        
        # Numeric range validation
        if isinstance(value, (int, float)):
            min_value = config_info.get('min_value')
            max_value = config_info.get('max_value')
            
            if min_value is not None and value < min_value:
                return False
            if max_value is not None and value > max_value:
                return False
        
        return True

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        
        config_value = self.configs.get(key)
        if config_value is None:
            return default
        
        # Log access to sensitive configuration
        if config_value.is_sensitive:
            audit_logger.log_security_event(
                event_type=SecurityEventType.DATA_ACCESS,
                severity=SecurityEventSeverity.LOW,
                message=f"Sensitive configuration accessed: {key}",
                details={'config_key': key, 'environment': self.environment}
            )
        
        return config_value.value

    def set(self, key: str, value: Any, is_sensitive: bool = False, persist: bool = False):
        """Set configuration value."""
        
        # Create config value
        config_value = ConfigValue(
            value=value,
            is_sensitive=is_sensitive,
            environment=self.environment
        )
        
        old_value = self.configs.get(key)
        self.configs[key] = config_value
        
        # Log configuration change
        audit_logger.log_security_event(
            event_type=SecurityEventType.SYSTEM_CONFIGURATION_CHANGE,
            severity=SecurityEventSeverity.MEDIUM if is_sensitive else SecurityEventSeverity.LOW,
            message=f"Configuration changed: {key}",
            details={
                'config_key': key,
                'has_old_value': old_value is not None,
                'is_sensitive': is_sensitive,
                'environment': self.environment
            }
        )
        
        # Persist to file if requested
        if persist:
            self._persist_config(key, config_value)
        
        # Notify change handlers
        self._notify_change_handlers(key, value, old_value.value if old_value else None)

    def _persist_config(self, key: str, config_value: ConfigValue):
        """Persist configuration to file."""
        
        if config_value.is_sensitive:
            # Save to encrypted file
            self._save_encrypted_config(key, config_value)
        else:
            # Save to regular config file
            self._save_regular_config(key, config_value)

    def _save_encrypted_config(self, key: str, config_value: ConfigValue):
        """Save sensitive configuration to encrypted file."""
        
        secrets_file = self.config_dir / f'secrets.{self.environment}.encrypted'
        
        # Load existing secrets
        existing_secrets = {}
        if secrets_file.exists():
            try:
                with open(secrets_file, 'rb') as f:
                    encrypted_data = f.read()
                decrypted_data = self.cipher_suite.decrypt(encrypted_data)
                existing_secrets = json.loads(decrypted_data.decode())
            except Exception:
                pass
        
        # Update with new value
        existing_secrets[key] = config_value.value
        
        # Encrypt and save
        data_json = json.dumps(existing_secrets)
        encrypted_data = self.cipher_suite.encrypt(data_json.encode())
        
        self.config_dir.mkdir(exist_ok=True)
        with open(secrets_file, 'wb') as f:
            f.write(encrypted_data)
        
        # Set restrictive permissions
        os.chmod(secrets_file, 0o600)

    def _save_regular_config(self, key: str, config_value: ConfigValue):
        """Save regular configuration to JSON file."""
        
        config_file = self.config_dir / f'{self.environment}.json'
        
        # Load existing configuration
        existing_config = {}
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    existing_config = json.load(f)
            except Exception:
                pass
        
        # Update with new value
        existing_config[key] = {
            'value': config_value.value,
            'sensitive': config_value.is_sensitive,
            'environment': config_value.environment,
            'description': config_value.description,
            'version': config_value.version
        }
        
        # Save configuration
        self.config_dir.mkdir(exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(existing_config, f, indent=2)

    def get_all_configs(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Get all configuration values."""
        
        result = {}
        for key, config_value in self.configs.items():
            if config_value.is_sensitive and not include_sensitive:
                result[key] = '[SENSITIVE]'
            else:
                result[key] = config_value.value
        
        return result

    def reload_configuration(self):
        """Reload configuration from files."""
        
        old_configs = self.configs.copy()
        self.configs.clear()
        
        try:
            self._load_configurations()
            
            audit_logger.log_security_event(
                event_type=SecurityEventType.SYSTEM_CONFIGURATION_CHANGE,
                severity=SecurityEventSeverity.LOW,
                message="Configuration reloaded successfully",
                details={'environment': self.environment}
            )
            
        except Exception as e:
            # Restore old configuration on failure
            self.configs = old_configs
            
            audit_logger.log_security_event(
                event_type=SecurityEventType.SYSTEM_CONFIGURATION_CHANGE,
                severity=SecurityEventSeverity.HIGH,
                message="Configuration reload failed",
                details={'error': str(e), 'environment': self.environment}
            )
            raise

    def add_change_handler(self, handler):
        """Add configuration change handler."""
        self.change_handlers.append(handler)

    def _notify_change_handlers(self, key: str, new_value: Any, old_value: Any):
        """Notify all change handlers of configuration change."""
        
        for handler in self.change_handlers:
            try:
                handler(key, new_value, old_value)
            except Exception as e:
                audit_logger.log_security_event(
                    event_type=SecurityEventType.SYSTEM_CONFIGURATION_CHANGE,
                    severity=SecurityEventSeverity.LOW,
                    message=f"Configuration change handler failed: {str(e)}",
                    details={'config_key': key, 'handler': str(handler)}
                )

    def export_configuration(self, include_sensitive: bool = False) -> str:
        """Export configuration for backup or migration."""
        
        config_data = {}
        for key, config_value in self.configs.items():
            if config_value.is_sensitive and not include_sensitive:
                continue
            
            config_data[key] = {
                'value': config_value.value,
                'encrypted': config_value.is_encrypted,
                'sensitive': config_value.is_sensitive,
                'environment': config_value.environment,
                'description': config_value.description,
                'version': config_value.version
            }
        
        return json.dumps(config_data, indent=2)

    def import_configuration(self, config_json: str, merge: bool = True):
        """Import configuration from JSON."""
        
        try:
            config_data = json.loads(config_json)
            
            if not merge:
                self.configs.clear()
            
            for key, config_info in config_data.items():
                config_value = ConfigValue(
                    value=config_info['value'],
                    is_encrypted=config_info.get('encrypted', False),
                    is_sensitive=config_info.get('sensitive', False),
                    environment=config_info.get('environment', 'all'),
                    description=config_info.get('description', ''),
                    version=config_info.get('version', '1.0')
                )
                
                self.configs[key] = config_value
            
            audit_logger.log_security_event(
                event_type=SecurityEventType.SYSTEM_CONFIGURATION_CHANGE,
                severity=SecurityEventSeverity.MEDIUM,
                message="Configuration imported",
                details={
                    'config_count': len(config_data),
                    'merge': merge,
                    'environment': self.environment
                }
            )
            
        except Exception as e:
            audit_logger.log_security_event(
                event_type=SecurityEventType.SYSTEM_CONFIGURATION_CHANGE,
                severity=SecurityEventSeverity.HIGH,
                message="Configuration import failed",
                details={'error': str(e)}
            )
            raise


class SecretsManager:
    """
    Dedicated secrets management with rotation capabilities.
    """

    def __init__(self, config_manager: SecureConfigManager):
        self.config_manager = config_manager
        self.rotation_schedule = {}

    def store_secret(self, key: str, value: str, rotate_days: int = None):
        """Store a secret with optional rotation schedule."""
        
        # Store the secret
        self.config_manager.set(key, value, is_sensitive=True, persist=True)
        
        # Set rotation schedule if specified
        if rotate_days:
            from datetime import datetime, timedelta
            next_rotation = datetime.now() + timedelta(days=rotate_days)
            self.rotation_schedule[key] = {
                'next_rotation': next_rotation.isoformat(),
                'rotation_days': rotate_days
            }
        
        audit_logger.log_security_event(
            event_type=SecurityEventType.SYSTEM_CONFIGURATION_CHANGE,
            severity=SecurityEventSeverity.MEDIUM,
            message=f"Secret stored: {key}",
            details={'key': key, 'has_rotation': rotate_days is not None}
        )

    def rotate_secret(self, key: str) -> str:
        """Rotate a secret and return the new value."""
        
        # Generate new secret
        new_secret = security_manager.generate_secure_token(32)
        
        # Store old secret for transition period
        old_secret = self.config_manager.get(key)
        if old_secret:
            self.config_manager.set(f"{key}_old", old_secret, is_sensitive=True)
        
        # Store new secret
        self.store_secret(key, new_secret)
        
        audit_logger.log_security_event(
            event_type=SecurityEventType.SYSTEM_CONFIGURATION_CHANGE,
            severity=SecurityEventSeverity.HIGH,
            message=f"Secret rotated: {key}",
            details={'key': key}
        )
        
        return new_secret

    def check_rotation_schedule(self):
        """Check and perform scheduled secret rotations."""
        
        from datetime import datetime
        now = datetime.now()
        
        for key, schedule_info in self.rotation_schedule.items():
            next_rotation = datetime.fromisoformat(schedule_info['next_rotation'])
            
            if now >= next_rotation:
                self.rotate_secret(key)

    def get_secret(self, key: str) -> Optional[str]:
        """Get a secret value."""
        return self.config_manager.get(key)

    def delete_secret(self, key: str):
        """Securely delete a secret."""
        
        if key in self.config_manager.configs:
            del self.config_manager.configs[key]
            
            # Remove from rotation schedule
            if key in self.rotation_schedule:
                del self.rotation_schedule[key]
            
            audit_logger.log_security_event(
                event_type=SecurityEventType.SYSTEM_CONFIGURATION_CHANGE,
                severity=SecurityEventSeverity.MEDIUM,
                message=f"Secret deleted: {key}",
                details={'key': key}
            )


class EnvironmentManager:
    """
    Manage environment-specific configurations and deployments.
    """

    def __init__(self):
        self.environments = {
            'development': {
                'debug': True,
                'log_level': 'DEBUG',
                'allowed_hosts': ['localhost', '127.0.0.1'],
                'cors_origins': ['http://localhost:3000'],
                'security_level': 'medium'
            },
            'testing': {
                'debug': False,
                'log_level': 'INFO',
                'allowed_hosts': ['test.fraud-platform.com'],
                'cors_origins': ['https://test-app.fraud-platform.com'],
                'security_level': 'high'
            },
            'staging': {
                'debug': False,
                'log_level': 'INFO',
                'allowed_hosts': ['staging.fraud-platform.com'],
                'cors_origins': ['https://staging-app.fraud-platform.com'],
                'security_level': 'high'
            },
            'production': {
                'debug': False,
                'log_level': 'WARNING',
                'allowed_hosts': ['api.fraud-platform.com'],
                'cors_origins': ['https://app.fraud-platform.com'],
                'security_level': 'maximum'
            }
        }

    def get_environment_config(self, environment: str) -> Dict[str, Any]:
        """Get configuration for specific environment."""
        return self.environments.get(environment, {})

    def validate_environment_transition(
        self, 
        from_env: str, 
        to_env: str
    ) -> List[str]:
        """Validate environment transition and return any issues."""
        
        issues = []
        
        # Check if moving to production with debug enabled
        if to_env == 'production':
            if self.environments[to_env].get('debug', False):
                issues.append("Debug mode should be disabled in production")
        
        # Check security level progression
        security_levels = {'low': 1, 'medium': 2, 'high': 3, 'maximum': 4}
        from_level = security_levels.get(
            self.environments.get(from_env, {}).get('security_level', 'medium'), 2
        )
        to_level = security_levels.get(
            self.environments.get(to_env, {}).get('security_level', 'medium'), 2
        )
        
        if to_level < from_level:
            issues.append("Security level should not decrease when moving to higher environment")
        
        return issues


# Initialize configuration management components
config_manager = SecureConfigManager()
secrets_manager = SecretsManager(config_manager)
environment_manager = EnvironmentManager()