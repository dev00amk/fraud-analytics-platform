from django.apps import AppConfig


class AlertsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.alerts'
    verbose_name = 'Fraud Alerts'

    def ready(self):
        """Import signal handlers when the app is ready."""
        try:
            import apps.alerts.signals  # noqa F401
        except ImportError:
            pass