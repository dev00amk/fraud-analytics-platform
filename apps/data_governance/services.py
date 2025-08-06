"""
Enterprise Data Governance Services
Implements GDPR, CCPA, PCI DSS compliance automation.
"""

import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import ValidationError
from django.db import transaction
from django.utils import timezone

from .models import (
    DataAccessLog,
    DataAsset,
    DataLineage,
    DataProcessingActivity,
    DataRetentionPolicy,
    DataSubject,
    PrivacyImpactAssessment,
    ProcessingPurpose,
)

User = get_user_model()
logger = logging.getLogger(__name__)


class GDPRComplianceService:
    """GDPR compliance automation service."""
    
    def __init__(self):
        self.retention_periods = {
            'transaction_data': 2555,  # 7 years for financial records
            'user_data': 1095,  # 3 years for user profiles
            'audit_logs': 2555,  # 7 years for audit trails
            'marketing_data': 365,  # 1 year for marketing
        }
    
    def register_data_subject(self, external_id: str, email: str = "", 
                            phone: str = "") -> DataSubject:
        """Register a new data subject for GDPR compliance."""
        try:
            data_subject, created = DataSubject.objects.get_or_create(
                external_id=external_id,
                defaults={
                    'email': email,
                    'phone': phone,
                    'consent_given': False,
                }
            )
            
            if created:
                logger.info(f"Registered new data subject: {external_id}")
            
            return data_subject
            
        except Exception as e:
            logger.error(f"Failed to register data subject {external_id}: {e}")
            raise
    
    def record_consent(self, data_subject_id: str, consent_given: bool,
                      consent_details: Dict[str, Any]) -> bool:
        """Record consent for data processing."""
        try:
            data_subject = DataSubject.objects.get(external_id=data_subject_id)
            
            data_subject.consent_given = consent_given
            data_subject.consent_date = timezone.now() if consent_given else None
            data_subject.consent_withdrawn = not consent_given
            data_subject.consent_withdrawal_date = timezone.now() if not consent_given else None
            data_subject.save()
            
            # Log the consent action
            self._log_data_access(
                user=None,
                data_subject=data_subject,
                content_object=data_subject,
                access_type='CONSENT_UPDATE',
                purpose=ProcessingPurpose.CONSENT,
                ip_address=consent_details.get('ip_address', '127.0.0.1'),
                user_agent=consent_details.get('user_agent', 'System'),
            )
            
            logger.info(f"Recorded consent for data subject {data_subject_id}: {consent_given}")
            return True
            
        except DataSubject.DoesNotExist:
            logger.error(f"Data subject not found: {data_subject_id}")
            return False
        except Exception as e:
            logger.error(f"Failed to record consent for {data_subject_id}: {e}")
            return False
    
    def handle_right_to_access(self, data_subject_id: str, 
                              requester_ip: str) -> Dict[str, Any]:
        """Handle GDPR Article 15 - Right of Access."""
        try:
            data_subject = DataSubject.objects.get(external_id=data_subject_id)
            data_subject.right_to_access_requested = True
            data_subject.save()
            
            # Collect all data for this subject
            personal_data = self._collect_personal_data(data_subject)
            
            # Log the access request
            self._log_data_access(
                user=None,
                data_subject=data_subject,
                content_object=data_subject,
                access_type='RIGHT_TO_ACCESS',
                purpose=ProcessingPurpose.LEGAL_OBLIGATION,
                ip_address=requester_ip,
                user_agent='GDPR Request',
            )
            
            logger.info(f"Processed right to access for {data_subject_id}")
            
            return {
                'status': 'success',
                'data_subject_id': data_subject_id,
                'personal_data': personal_data,
                'processing_activities': self._get_processing_activities(data_subject),
                'retention_periods': self._get_retention_info(data_subject),
                'request_timestamp': timezone.now().isoformat(),
            }
            
        except DataSubject.DoesNotExist:
            logger.error(f"Data subject not found for access request: {data_subject_id}")
            return {'status': 'error', 'message': 'Data subject not found'}
        except Exception as e:
            logger.error(f"Failed to process right to access for {data_subject_id}: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def handle_right_to_erasure(self, data_subject_id: str, 
                               requester_ip: str) -> Dict[str, Any]:
        """Handle GDPR Article 17 - Right to Erasure (Right to be Forgotten)."""
        try:
            data_subject = DataSubject.objects.get(external_id=data_subject_id)
            
            # Check if erasure is legally possible
            legal_obligations = self._check_legal_obligations(data_subject)
            if legal_obligations:
                return {
                    'status': 'denied',
                    'reason': 'Legal obligations prevent erasure',
                    'details': legal_obligations
                }
            
            # Perform erasure
            with transaction.atomic():
                data_subject.right_to_erasure_requested = True
                data_subject.save()
                
                # Anonymize or delete personal data
                erasure_results = self._perform_data_erasure(data_subject)
                
                # Log the erasure
                self._log_data_access(
                    user=None,
                    data_subject=data_subject,
                    content_object=data_subject,
                    access_type='RIGHT_TO_ERASURE',
                    purpose=ProcessingPurpose.LEGAL_OBLIGATION,
                    ip_address=requester_ip,
                    user_agent='GDPR Request',
                )
            
            logger.info(f"Processed right to erasure for {data_subject_id}")
            
            return {
                'status': 'success',
                'data_subject_id': data_subject_id,
                'erasure_results': erasure_results,
                'request_timestamp': timezone.now().isoformat(),
            }
            
        except DataSubject.DoesNotExist:
            logger.error(f"Data subject not found for erasure request: {data_subject_id}")
            return {'status': 'error', 'message': 'Data subject not found'}
        except Exception as e:
            logger.error(f"Failed to process right to erasure for {data_subject_id}: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def handle_data_portability(self, data_subject_id: str, 
                               export_format: str = 'json') -> Dict[str, Any]:
        """Handle GDPR Article 20 - Right to Data Portability."""
        try:
            data_subject = DataSubject.objects.get(external_id=data_subject_id)
            data_subject.right_to_portability_requested = True
            data_subject.save()
            
            # Export data in machine-readable format
            portable_data = self._export_portable_data(data_subject, export_format)
            
            logger.info(f"Generated portable data export for {data_subject_id}")
            
            return {
                'status': 'success',
                'data_subject_id': data_subject_id,
                'export_format': export_format,
                'data': portable_data,
                'export_timestamp': timezone.now().isoformat(),
            }
            
        except DataSubject.DoesNotExist:
            logger.error(f"Data subject not found for portability request: {data_subject_id}")
            return {'status': 'error', 'message': 'Data subject not found'}
        except Exception as e:
            logger.error(f"Failed to process data portability for {data_subject_id}: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _collect_personal_data(self, data_subject: DataSubject) -> Dict[str, Any]:
        """Collect all personal data for a data subject."""
        from apps.transactions.models import Transaction
        from apps.cases.models import Case
        
        personal_data = {
            'profile': {
                'external_id': data_subject.external_id,
                'email': data_subject.email,
                'phone': data_subject.phone,
                'consent_status': data_subject.consent_given,
                'consent_date': data_subject.consent_date.isoformat() if data_subject.consent_date else None,
            },
            'transactions': [],
            'cases': [],
            'access_logs': [],
        }
        
        # Collect transaction data
        transactions = Transaction.objects.filter(user_id=data_subject.external_id)
        for transaction in transactions:
            personal_data['transactions'].append({
                'transaction_id': transaction.transaction_id,
                'amount': str(transaction.amount),
                'currency': transaction.currency,
                'timestamp': transaction.timestamp.isoformat(),
                'status': transaction.status,
            })
        
        # Collect case data
        cases = Case.objects.filter(transaction_id__in=[t.transaction_id for t in transactions])
        for case in cases:
            personal_data['cases'].append({
                'case_number': case.case_number,
                'title': case.title,
                'status': case.status,
                'created_at': case.created_at.isoformat(),
            })
        
        # Collect access logs
        access_logs = DataAccessLog.objects.filter(data_subject=data_subject)[:100]  # Last 100 entries
        for log in access_logs:
            personal_data['access_logs'].append({
                'access_type': log.access_type,
                'timestamp': log.timestamp.isoformat(),
                'purpose': log.purpose,
                'ip_address': log.ip_address,
            })
        
        return personal_data
    
    def _get_processing_activities(self, data_subject: DataSubject) -> List[Dict[str, Any]]:
        """Get processing activities involving this data subject."""
        activities = DataProcessingActivity.objects.filter(
            assets__in=DataAsset.objects.filter(contains_pii=True)
        ).distinct()
        
        return [
            {
                'name': activity.name,
                'purpose': activity.purpose,
                'lawful_basis': activity.lawful_basis,
                'retention_period': activity.retention_period,
                'controller': activity.controller_name,
            }
            for activity in activities
        ]
    
    def _get_retention_info(self, data_subject: DataSubject) -> Dict[str, Any]:
        """Get data retention information."""
        return {
            'transaction_data': f"{self.retention_periods['transaction_data']} days",
            'user_data': f"{self.retention_periods['user_data']} days",
            'audit_logs': f"{self.retention_periods['audit_logs']} days",
        }
    
    def _check_legal_obligations(self, data_subject: DataSubject) -> List[str]:
        """Check if there are legal obligations preventing erasure."""
        obligations = []
        
        # Check for ongoing fraud investigations
        from apps.cases.models import Case
        open_cases = Case.objects.filter(
            transaction_id__in=Transaction.objects.filter(
                user_id=data_subject.external_id
            ).values_list('transaction_id', flat=True),
            status__in=['open', 'investigating']
        )
        
        if open_cases.exists():
            obligations.append("Ongoing fraud investigation cases")
        
        # Check for regulatory retention requirements
        recent_transactions = Transaction.objects.filter(
            user_id=data_subject.external_id,
            created_at__gte=timezone.now() - timedelta(days=2555)  # 7 years
        )
        
        if recent_transactions.exists():
            obligations.append("Financial record retention requirements (7 years)")
        
        return obligations
    
    def _perform_data_erasure(self, data_subject: DataSubject) -> Dict[str, Any]:
        """Perform actual data erasure/anonymization."""
        from apps.transactions.models import Transaction
        
        results = {
            'transactions_anonymized': 0,
            'profile_data_erased': False,
            'access_logs_retained': True,  # Keep for audit purposes
        }
        
        # Anonymize transaction data (keep for fraud detection but remove PII)
        transactions = Transaction.objects.filter(user_id=data_subject.external_id)
        for transaction in transactions:
            # Replace user_id with anonymized hash
            anonymous_id = hashlib.sha256(
                f"anonymous_{transaction.id}_{timezone.now().timestamp()}".encode()
            ).hexdigest()[:16]
            
            transaction.user_id = f"anon_{anonymous_id}"
            transaction.save()
            
            results['transactions_anonymized'] += 1
        
        # Erase profile data but keep the record for audit
        data_subject.email = ""
        data_subject.phone = ""
        data_subject.save()
        results['profile_data_erased'] = True
        
        return results
    
    def _export_portable_data(self, data_subject: DataSubject, 
                             export_format: str) -> Dict[str, Any]:
        """Export data in portable format."""
        portable_data = self._collect_personal_data(data_subject)
        
        # Add metadata
        portable_data['export_metadata'] = {
            'export_date': timezone.now().isoformat(),
            'format': export_format,
            'version': '1.0',
            'data_controller': settings.COMPANY_NAME if hasattr(settings, 'COMPANY_NAME') else 'Fraud Analytics Platform',
        }
        
        return portable_data
    
    def _log_data_access(self, user: Optional[User], data_subject: Optional[DataSubject],
                        content_object: Any, access_type: str, purpose: str,
                        ip_address: str, user_agent: str, **kwargs) -> DataAccessLog:
        """Log data access for audit purposes."""
        content_type = ContentType.objects.get_for_model(content_object)
        
        return DataAccessLog.objects.create(
            user=user,
            data_subject=data_subject,
            content_type=content_type,
            object_id=content_object.pk,
            access_type=access_type,
            purpose=purpose,
            ip_address=ip_address,
            user_agent=user_agent,
            response_status=kwargs.get('response_status', 200),
            records_accessed=kwargs.get('records_accessed', 1),
            data_exported=kwargs.get('data_exported', False),
        )


class DataRetentionService:
    """Automated data retention and deletion service."""
    
    def __init__(self):
        self.dry_run = False
    
    def enforce_retention_policies(self, dry_run: bool = False) -> Dict[str, Any]:
        """Enforce all active retention policies."""
        self.dry_run = dry_run
        results = {
            'policies_processed': 0,
            'records_deleted': 0,
            'errors': [],
            'dry_run': dry_run,
        }
        
        policies = DataRetentionPolicy.objects.filter(
            auto_delete_enabled=True,
            legal_hold_enabled=False
        )
        
        for policy in policies:
            try:
                policy_results = self._enforce_single_policy(policy)
                results['policies_processed'] += 1
                results['records_deleted'] += policy_results['deleted_count']
                
                logger.info(f"Processed retention policy {policy.name}: "
                          f"{policy_results['deleted_count']} records deleted")
                
            except Exception as e:
                error_msg = f"Failed to process policy {policy.name}: {str(e)}"
                results['errors'].append(error_msg)
                logger.error(error_msg)
        
        return results
    
    def _enforce_single_policy(self, policy: DataRetentionPolicy) -> Dict[str, Any]:
        """Enforce a single retention policy."""
        cutoff_date = timezone.now() - timedelta(days=policy.retention_period_days)
        
        # Get the model class for the asset
        model_class = self._get_model_class(policy.asset)
        if not model_class:
            raise ValueError(f"Cannot find model for asset {policy.asset.name}")
        
        # Find records to delete
        queryset = model_class.objects.filter(created_at__lt=cutoff_date)
        
        if policy.deletion_method == 'soft_delete':
            # Soft delete (mark as deleted)
            if not self.dry_run:
                deleted_count = queryset.update(deleted_at=timezone.now())
            else:
                deleted_count = queryset.count()
        else:
            # Hard delete
            if not self.dry_run:
                deleted_count, _ = queryset.delete()
            else:
                deleted_count = queryset.count()
        
        return {'deleted_count': deleted_count}
    
    def _get_model_class(self, asset: DataAsset):
        """Get Django model class from asset definition."""
        # This would map asset definitions to actual Django models
        model_mapping = {
            'transactions': 'apps.transactions.models.Transaction',
            'cases': 'apps.cases.models.Case',
            'access_logs': 'apps.data_governance.models.DataAccessLog',
        }
        
        model_path = model_mapping.get(asset.table_name)
        if not model_path:
            return None
        
        # Dynamically import the model
        module_path, class_name = model_path.rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)


class DataLineageService:
    """Data lineage tracking for compliance and debugging."""
    
    def track_data_transformation(self, source_asset: DataAsset, 
                                 destination_asset: DataAsset,
                                 transformation_details: Dict[str, Any],
                                 processing_activity: DataProcessingActivity) -> DataLineage:
        """Track a data transformation for lineage."""
        return DataLineage.objects.create(
            source_asset=source_asset,
            destination_asset=destination_asset,
            transformation_type=transformation_details.get('type', 'unknown'),
            transformation_logic=transformation_details.get('logic', ''),
            transformation_code=transformation_details.get('code', ''),
            processing_activity=processing_activity,
            processor=transformation_details.get('processor', 'system'),
            data_quality_score=transformation_details.get('quality_score'),
            completeness_score=transformation_details.get('completeness_score'),
            accuracy_score=transformation_details.get('accuracy_score'),
        )
    
    def get_data_lineage(self, asset: DataAsset, depth: int = 3) -> Dict[str, Any]:
        """Get complete data lineage for an asset."""
        lineage = {
            'asset': {
                'name': asset.name,
                'classification': asset.classification,
                'contains_pii': asset.contains_pii,
            },
            'upstream': self._get_upstream_lineage(asset, depth),
            'downstream': self._get_downstream_lineage(asset, depth),
        }
        
        return lineage
    
    def _get_upstream_lineage(self, asset: DataAsset, depth: int) -> List[Dict[str, Any]]:
        """Get upstream data lineage (sources)."""
        if depth <= 0:
            return []
        
        upstream = []
        lineage_records = DataLineage.objects.filter(destination_asset=asset)
        
        for record in lineage_records:
            upstream.append({
                'asset': {
                    'name': record.source_asset.name,
                    'classification': record.source_asset.classification,
                },
                'transformation': {
                    'type': record.transformation_type,
                    'logic': record.transformation_logic,
                    'quality_score': record.data_quality_score,
                },
                'upstream': self._get_upstream_lineage(record.source_asset, depth - 1),
            })
        
        return upstream
    
    def _get_downstream_lineage(self, asset: DataAsset, depth: int) -> List[Dict[str, Any]]:
        """Get downstream data lineage (destinations)."""
        if depth <= 0:
            return []
        
        downstream = []
        lineage_records = DataLineage.objects.filter(source_asset=asset)
        
        for record in lineage_records:
            downstream.append({
                'asset': {
                    'name': record.destination_asset.name,
                    'classification': record.destination_asset.classification,
                },
                'transformation': {
                    'type': record.transformation_type,
                    'logic': record.transformation_logic,
                    'quality_score': record.data_quality_score,
                },
                'downstream': self._get_downstream_lineage(record.destination_asset, depth - 1),
            })
        
        return downstream