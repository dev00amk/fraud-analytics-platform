"""
Django settings for fraud_platform project.
"""

import os
from pathlib import Path

import environ

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Environment variables
env = environ.Env(DEBUG=(bool, False))

# Take environment variables from .env file
environ.Env.read_env(BASE_DIR / ".env")

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = env("SECRET_KEY", default="django-insecure-change-me-in-production")

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = env("DEBUG")

ALLOWED_HOSTS = env.list("ALLOWED_HOSTS", default=["localhost", "127.0.0.1"])

# Application definition
DJANGO_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
]

THIRD_PARTY_APPS = [
    "rest_framework",
    "rest_framework_simplejwt",
    "corsheaders",
    "drf_spectacular",
]

LOCAL_APPS = [
    "apps.authentication",
    "apps.transactions",
    "apps.fraud_detection",
    "apps.cases",
    "apps.analytics",
    "apps.webhooks",
    "apps.alerts",
]

INSTALLED_APPS = DJANGO_APPS + THIRD_PARTY_APPS + LOCAL_APPS

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "apps.security.api_security.APISecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "apps.security.error_handling.SecurityErrorMiddleware",
]

ROOT_URLCONF = "fraud_platform.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "fraud_platform.wsgi.application"

# Database
# Use SQLite for quick setup, PostgreSQL for production
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

# Uncomment below for PostgreSQL in production
# DATABASES = {"default": env.db()}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

# Internationalization
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_DIRS = []

# Media files
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

# Default primary key field type
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Custom user model
AUTH_USER_MODEL = "authentication.User"

# REST Framework configuration
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework_simplejwt.authentication.JWTAuthentication",
        "rest_framework.authentication.SessionAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",
    ],
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ],
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 20,
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
}

# JWT Configuration
from datetime import timedelta

SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": timedelta(minutes=60),
    "REFRESH_TOKEN_LIFETIME": timedelta(days=7),
    "ROTATE_REFRESH_TOKENS": True,
}

# API Documentation
SPECTACULAR_SETTINGS = {
    "TITLE": "Fraud Analytics Platform API",
    "DESCRIPTION": "Enterprise-grade fraud detection and transaction monitoring platform",
    "VERSION": "1.0.0",
    "SERVE_INCLUDE_SCHEMA": False,
}

# CORS Configuration
CORS_ALLOWED_ORIGINS = env.list(
    "CORS_ALLOWED_ORIGINS",
    default=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
)

# Security Framework Configuration
SECURITY_FRAMEWORK_ENABLED = env.bool("SECURITY_FRAMEWORK_ENABLED", default=True)

# Encryption keys for security framework
ENCRYPTION_KEY = env("ENCRYPTION_KEY", default=None)
PII_ENCRYPTION_KEY = env("PII_ENCRYPTION_KEY", default=None)
FINANCIAL_ENCRYPTION_KEY = env("FINANCIAL_ENCRYPTION_KEY", default=None)
SENSITIVE_ENCRYPTION_KEY = env("SENSITIVE_ENCRYPTION_KEY", default=None)
AUDIT_ENCRYPTION_KEY = env("AUDIT_ENCRYPTION_KEY", default=None)

# Security Middleware (add to MIDDLEWARE list)
SECURITY_MIDDLEWARE = [
    'apps.security.api_security.APISecurityMiddleware',
    'apps.security.error_handling.SecurityErrorMiddleware',
]

# Rate Limiting Configuration
RATE_LIMITING_ENABLED = env.bool("RATE_LIMITING_ENABLED", default=True)
RATE_LIMIT_DEFAULT_REQUESTS = env.int("RATE_LIMIT_DEFAULT_REQUESTS", default=1000)
RATE_LIMIT_DEFAULT_WINDOW = env.int("RATE_LIMIT_DEFAULT_WINDOW", default=3600)

# DDoS Protection
DDOS_PROTECTION_ENABLED = env.bool("DDOS_PROTECTION_ENABLED", default=True)
DDOS_THRESHOLD_CONNECTIONS = env.int("DDOS_THRESHOLD_CONNECTIONS", default=50)
DDOS_THRESHOLD_WINDOW = env.int("DDOS_THRESHOLD_WINDOW", default=10)

# Geographic Blocking
GEOGRAPHIC_BLOCKS = env.list("GEOGRAPHIC_BLOCKS", default=[])

# Blocked IPs
BLOCKED_IPS = env.list("BLOCKED_IPS", default=[])

# Audit and Compliance
AUDIT_LOGGING_ENABLED = env.bool("AUDIT_LOGGING_ENABLED", default=True)
COMPLIANCE_MONITORING_ENABLED = env.bool("COMPLIANCE_MONITORING_ENABLED", default=True)

# Security Alerting
SECURITY_ALERTS_ENABLED = env.bool("SECURITY_ALERTS_ENABLED", default=True)
SECURITY_ALERT_EMAIL = env("SECURITY_ALERT_EMAIL", default="security@fraud-platform.com")

# Configuration Management
CONFIG_ENCRYPTION_ENABLED = env.bool("CONFIG_ENCRYPTION_ENABLED", default=True)

# Celery Configuration
CELERY_BROKER_URL = env("CELERY_BROKER_URL", default="redis://localhost:6379/0")
CELERY_RESULT_BACKEND = env("CELERY_RESULT_BACKEND", default="redis://localhost:6379/0")
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_TIMEZONE = TIME_ZONE

# Redis Configuration
REDIS_URL = env("REDIS_URL", default="redis://localhost:6379/0")

# Cache Configuration
CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": REDIS_URL.replace('/0', '/1'),  # Use different DB for cache
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
        },
    }
}

# Fraud Detection Settings
FRAUD_THRESHOLD_LOW = env.int("FRAUD_THRESHOLD_LOW", default=30)
FRAUD_THRESHOLD_MEDIUM = env.int("FRAUD_THRESHOLD_MEDIUM", default=60)
FRAUD_THRESHOLD_HIGH = env.int("FRAUD_THRESHOLD_HIGH", default=80)

# Email Configuration
EMAIL_BACKEND = env(
    "EMAIL_BACKEND", default="django.core.mail.backends.console.EmailBackend"
)
EMAIL_HOST = env("EMAIL_HOST", default="")
EMAIL_PORT = env.int("EMAIL_PORT", default=587)
EMAIL_USE_TLS = env.bool("EMAIL_USE_TLS", default=True)
EMAIL_HOST_USER = env("EMAIL_HOST_USER", default="")
EMAIL_HOST_PASSWORD = env("EMAIL_HOST_PASSWORD", default="")

# Logging
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "{levelname} {asctime} {module} {process:d} {thread:d} {message}",
            "style": "{",
        },
    },
    "handlers": {
        "file": {
            "level": "INFO",
            "class": "logging.FileHandler",
            "filename": BASE_DIR / "logs" / "django.log",
            "formatter": "verbose",
        },
        "console": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "verbose",
        },
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "INFO",
    },
}

# Create logs directory if it doesn't exist
os.makedirs(BASE_DIR / "logs", exist_ok=True)
# Import datetime at the top
from datetime import timedelta
