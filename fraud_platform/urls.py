"""
URL configuration for fraud_platform project.
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView, SpectacularRedocView

urlpatterns = [
    # Admin
    path('admin/', admin.site.urls),
    
    # API Documentation
    path('api/schema/', SpectacularAPIView.as_view(), name='schema'),
    path('docs/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
    path('redoc/', SpectacularRedocView.as_view(url_name='schema'), name='redoc'),
    
    # API v1
    path('api/v1/auth/', include('apps.authentication.urls')),
    path('api/v1/transactions/', include('apps.transactions.urls')),
    path('api/v1/fraud/', include('apps.fraud_detection.urls')),
    path('api/v1/cases/', include('apps.cases.urls')),
    path('api/v1/analytics/', include('apps.analytics.urls')),
    path('api/v1/webhooks/', include('apps.webhooks.urls')),
    
    # Health check
    path('health/', include('apps.core.urls')),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)