from django.urls import path

from . import views

urlpatterns = [
    path("", views.CaseListCreateView.as_view(), name="case_list"),
    path("<uuid:pk>/", views.CaseDetailView.as_view(), name="case_detail"),
]
