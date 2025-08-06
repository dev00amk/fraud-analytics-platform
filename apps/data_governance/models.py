"""
Enterprise Data Governance Models
Implements GDPR, CCPA, PCI DSS compliance and data lineage tracking.
"""

import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

from django.contrib.auth import get_user_model
from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from django.db import models
from django.utils import timezone

User = get_user_model()


class DataClassification(models.TextChoices):
    """Data classification levels for governance."""
    PUBLIC = "public", "Public"
    INTERNAL = "internal", "Internal"
    CONFIDENTIAL = "confidential", "Confidential"
    RESTRICTED = "restricted", "Restricted"
    PII = "pii", "Personally Identifiable Information"
    FINANCIAL = "financial", "Financial Data"


class ProcessingPurpose(models.TextChoices):
    """GDPR Article 6 lawful basis for processing."""
    CONSENT = "consent", "Consent"
    CONTRACT = "contract", "Contract Performance"
    LEGAL_OBLIGATION = "legal_obligation", "Legal Obligation"
    VITAL_INTERESTS = "vital_interests", "Vital Interests"
    PUBLIC_TASK = "public_task", "Public Task"
    LEGITIMATE_INTERESTS = "legitimate_interests", "Legitimate Interests"
    FRAUD_PREVENTION = "fraud_prevention", "Fraud Prevention"


class DataSubject(models.Model):
    """GDPR Data Subject (individual whose data is processed)."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    external_id = models.CharField(max_length=255, unique=True)
    email = models.EmailField(blank=True)
    phone = models.CharField(max_length=20, blank=True)
    
    # Consent management
    consent_given = models.BooleanField(default=False)
    consent_date = models.DateTimeField(null=True, blank=True)
    consent_withdrawn = models.BooleanField(default=False)
    consent_withdrawal_date = models.DateTimeField(null=True, blank=True)
    
    # Rights exercised
    right_to_access_requested = models.BooleanField(default=False)
    right_to_rectification_requested = models.BooleanField(default=False)
    right_to_erasure_requested = models.BooleanField(default=False)
    right_to_portability_requested = models.BooleanField(default=False)
    right_to_object_requested = models.BooleanField(default=False)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'data_governance_subject'
        indexes = [
            models.Index(fields=['external_id']),
            models.Index(fields=['email']),
        ]


class DataAsset(models.Model):
    """Catalog of data assets for governance."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    description = models.TextField()
    
    # Classification
    classification = models.CharField(
        max_length=20,
        choices=DataClassification.choices,
        default=DataClassification.INTERNAL
    )
    
    # Data location and structure
    database_name = models.CharField(max_length=100)
    table_name = models.CharField(max_length=100)
    schema_definition = models.JSONField(default=dict)
    
    # Governance metadata
    data_owner = models.ForeignKey(User, on_delete=models.PROTECT, related_name='owned_assets')
    data_steward = models.ForeignKey(User, on_delete=models.PROTECT, related_name='stewarded_assets')
    
    # Retention policy
    retention_period_days = models.IntegerField(default=2555)  # 7 years default
    auto_delete_enabled = models.BooleanField(default=False)
    
    # Compliance flags
    contains_pii = models.BooleanField(default=False)
    contains_financial_data = models.BooleanField(default=False)
    pci_dss_scope = models.BooleanField(default=False)
    gdpr_scope = models.BooleanField(default=False)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'data_governance_asset'
        unique_together = ['database_name', 'table_name']


class DataProcessingActivity(models.Model):
    """GDPR Article 30 - Record of Processing Activities."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    description = models.TextField()
    
    # Controller information
    controller_name = models.CharField(max_length=255)
    controller_contact = models.EmailField()
    
    # Processing details
    purpose = models.CharField(max_length=50, choices=ProcessingPurpose.choices)
    lawful_basis = models.CharField(max_length=50, choices=ProcessingPurpose.choices)
    
    # Data categories
    data_categories = models.JSONField(default=list)  # List of data types
    data_subjects_categories = models.JSONField(default=list)  # Customer, employee, etc.
    
    # Recipients
    recipients = models.JSONField(default=list)  # Who receives the data
    third_country_transfers = models.JSONField(default=list)  # International transfers
    
    # Retention
    retention_period = models.CharField(max_length=255)
    deletion_schedule = models.DateTimeField(null=True, blank=True)
    
    # Security measures
    technical_measures = models.JSONField(default=list)
    organizational_measures = models.JSONField(default=list)
    
    # Assets involved
    assets = models.ManyToManyField(DataAsset, related_name='processing_activities')
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'data_governance_processing_activity'


class DataLineage(models.Model):
    """Track data lineage for compliance and debugging."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Source and destination
    source_asset = models.ForeignKey(DataAsset, on_delete=models.CASCADE, related_name='lineage_source')
    destination_asset = models.ForeignKey(DataAsset, on_delete=models.CASCADE, related_name='lineage_destination')
    
    # Transformation details
    transformation_type = models.CharField(max_length=50)  # ETL, API, etc.
    transformation_logic = models.TextField()
    transformation_code = models.TextField(blank=True)
    
    # Processing metadata
    processing_activity = models.ForeignKey(DataProcessingActivity, on_delete=models.CASCADE)
    processor = models.CharField(max_length=255)  # System/service name
    
    # Quality metrics
    data_quality_score = models.FloatField(null=True, blank=True)
    completeness_score = models.FloatField(null=True, blank=True)
    accuracy_score = models.FloatField(null=True, blank=True)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'data_governance_lineage'


class DataAccessLog(models.Model):
    """Audit log for data access (GDPR Article 30)."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Access details
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    data_subject = models.ForeignKey(DataSubject, on_delete=models.CASCADE, null=True, blank=True)
    
    # What was accessed
    content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE)
    object_id = models.PositiveIntegerField()
    content_object = GenericForeignKey('content_type', 'object_id')
    
    # Access metadata
    access_type = models.CharField(max_length=20)  # READ, WRITE, DELETE, EXPORT
    purpose = models.CharField(max_length=50, choices=ProcessingPurpose.choices)
    ip_address = models.GenericIPAddressField()
    user_agent = models.TextField()
    
    # Request details
    request_id = models.CharField(max_length=255, blank=True)
    api_endpoint = models.CharField(max_length=255, blank=True)
    query_parameters = models.JSONField(default=dict)
    
    # Response details
    response_status = models.IntegerField()
    records_accessed = models.IntegerField(default=0)
    data_exported = models.BooleanField(default=False)
    
    # Metadata
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'data_governance_access_log'
        indexes = [
            models.Index(fields=['user', 'timestamp']),
            models.Index(fields=['data_subject', 'timestamp']),
            models.Index(fields=['access_type', 'timestamp']),
        ]


class DataRetentionPolicy(models.Model):
    """Data retention policies for compliance."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    description = models.TextField()
    
    # Policy details
    asset = models.ForeignKey(DataAsset, on_delete=models.CASCADE)
    retention_period_days = models.IntegerField()
    
    # Deletion rules
    auto_delete_enabled = models.BooleanField(default=True)
    deletion_method = models.CharField(max_length=50, default='soft_delete')
    
    # Legal hold
    legal_hold_enabled = models.BooleanField(default=False)
    legal_hold_reason = models.TextField(blank=True)
    legal_hold_expiry = models.DateTimeField(null=True, blank=True)
    
    # Compliance requirements
    regulatory_requirement = models.CharField(max_length=100, blank=True)  # GDPR, CCPA, etc.
    business_justification = models.TextField()
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(User, on_delete=models.PROTECT)
    
    class Meta:
        db_table = 'data_governance_retention_policy'


class PrivacyImpactAssessment(models.Model):
    """GDPR Article 35 - Data Protection Impact Assessment."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    description = models.TextField()
    
    # Assessment details
    processing_activity = models.ForeignKey(DataProcessingActivity, on_delete=models.CASCADE)
    risk_level = models.CharField(max_length=20, choices=[
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('very_high', 'Very High'),
    ])
    
    # Risk assessment
    identified_risks = models.JSONField(default=list)
    mitigation_measures = models.JSONField(default=list)
    residual_risk_level = models.CharField(max_length=20, choices=[
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('very_high', 'Very High'),
    ])
    
    # Approval
    approved = models.BooleanField(default=False)
    approved_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    approval_date = models.DateTimeField(null=True, blank=True)
    
    # Review schedule
    next_review_date = models.DateTimeField()
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(User, on_delete=models.PROTECT, related_name='created_pias')
    
    class Meta:
        db_table = 'data_governance_pia'