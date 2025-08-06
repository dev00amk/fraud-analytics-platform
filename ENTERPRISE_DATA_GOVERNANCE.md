# 📊 Enterprise Data Governance & Privacy Framework

## Executive Summary

This document outlines FraudGuard's comprehensive data governance framework ensuring data quality, privacy, security, and compliance across all data lifecycle stages. Our approach encompasses data classification, lineage tracking, privacy engineering, and automated compliance monitoring.

---

## 🎯 Data Governance Strategy

### **Core Principles**
1. **Data as an Asset**: Treat data as a strategic business asset with defined ownership
2. **Privacy by Design**: Embed privacy controls at the data architecture level
3. **Data Quality First**: Ensure data accuracy, completeness, and consistency
4. **Lineage Transparency**: Track data flow from source to consumption
5. **Automated Compliance**: Implement automated controls for regulatory requirements
6. **Principle of Least Privilege**: Minimize data access to required personnel only
7. **Data Minimization**: Collect and retain only necessary data

### **Governance Structure**
```yaml
data_governance_organization:
  data_governance_council:
    chair: "Chief Data Officer"
    members:
      - "Chief Technology Officer"
      - "Chief Information Security Officer"
      - "Chief Privacy Officer"
      - "Chief Compliance Officer"
      - "Head of Engineering"
      - "Head of Product"
    
    responsibilities:
      - "Define data strategy and policies"
      - "Approve data classification schemes"
      - "Review and approve data sharing agreements"
      - "Oversight of data breach incidents"
  
  data_stewards:
    fraud_detection_steward:
      role: "Senior ML Engineer"
      responsibilities:
        - "Ensure fraud model data quality"
        - "Manage feature engineering pipeline"
        - "Monitor model performance metrics"
    
    customer_data_steward:
      role: "Senior Backend Engineer"
      responsibilities:
        - "Manage customer PII data lifecycle"
        - "Implement privacy controls"
        - "Handle data subject rights requests"
    
    transaction_data_steward:
      role: "Senior Data Engineer"
      responsibilities:
        - "Ensure transaction data integrity"
        - "Manage data retention policies"
        - "Implement audit trail requirements"
  
  data_protection_officer:
    role: "Chief Privacy Officer"
    responsibilities:
      - "GDPR compliance oversight"
      - "Privacy impact assessments"
      - "Data breach notifications"
      - "Privacy training programs"
```

---

## 🏷️ Data Classification Framework

### **Classification Levels**
```python
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta

class DataClassification(Enum):
    """Enterprise data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    
    @property
    def protection_requirements(self) -> Dict[str, str]:
        """Get protection requirements for classification level."""
        requirements = {
            self.PUBLIC: {
                "encryption_at_rest": "optional",
                "encryption_in_transit": "optional",
                "access_logging": "optional",
                "retention_period": "indefinite"
            },
            self.INTERNAL: {
                "encryption_at_rest": "required",
                "encryption_in_transit": "required",
                "access_logging": "required",
                "retention_period": "7 years"
            },
            self.CONFIDENTIAL: {
                "encryption_at_rest": "AES-256",
                "encryption_in_transit": "TLS 1.3",
                "access_logging": "required",
                "access_approval": "required",
                "retention_period": "as per policy"
            },
            self.RESTRICTED: {
                "encryption_at_rest": "AES-256 + HSM",
                "encryption_in_transit": "mTLS",
                "access_logging": "required",
                "access_approval": "dual approval required",
                "data_masking": "required",
                "retention_period": "as per policy",
                "geographic_restrictions": "required"
            }
        }
        return requirements.get(self, {})

@dataclass
class DataElement:
    """Represents a classified data element."""
    name: str
    classification: DataClassification
    data_type: str
    description: str
    owner: str
    steward: str
    source_system: str
    created_date: datetime
    last_updated: datetime
    retention_period: Optional[timedelta] = None
    geographic_restrictions: Optional[List[str]] = None
    processing_purpose: Optional[str] = None
    legal_basis: Optional[str] = None  # For GDPR compliance
    
    def get_retention_date(self) -> Optional[datetime]:
        """Calculate data retention expiration date."""
        if self.retention_period:
            return self.created_date + self.retention_period
        return None
    
    def is_expired(self) -> bool:
        """Check if data has exceeded retention period."""
        retention_date = self.get_retention_date()
        if retention_date:
            return datetime.now() > retention_date
        return False

class DataClassificationEngine:
    """Automated data classification engine."""
    
    def __init__(self):
        self.classification_rules = self._load_classification_rules()
        self.pii_patterns = self._load_pii_patterns()
    
    def classify_data_element(self, element_name: str, sample_values: List[str]) -> DataClassification:
        """Automatically classify data element based on name and content."""
        
        # Check for PII patterns
        if self._contains_pii(element_name, sample_values):
            return DataClassification.RESTRICTED
        
        # Check for financial data
        if self._is_financial_data(element_name, sample_values):
            return DataClassification.CONFIDENTIAL
        
        # Check for internal business data
        if self._is_internal_data(element_name):
            return DataClassification.INTERNAL
        
        # Default to public if no sensitive patterns detected
        return DataClassification.PUBLIC
    
    def _contains_pii(self, element_name: str, sample_values: List[str]) -> bool:
        """Check if data contains personally identifiable information."""
        pii_indicators = [
            'email', 'phone', 'ssn', 'social_security', 'passport', 'license',
            'address', 'name', 'dob', 'birth_date', 'credit_card', 'account_number'
        ]
        
        element_lower = element_name.lower()
        
        # Check element name
        if any(indicator in element_lower for indicator in pii_indicators):
            return True
        
        # Check sample values against regex patterns
        for value in sample_values[:10]:  # Check first 10 samples
            if self._matches_pii_pattern(value):
                return True
        
        return False
    
    def _matches_pii_pattern(self, value: str) -> bool:
        """Check if value matches PII regex patterns."""
        import re
        
        patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$',
            'ssn': r'^\d{3}-?\d{2}-?\d{4}$',
            'credit_card': r'^\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}$'
        }
        
        return any(re.match(pattern, str(value)) for pattern in patterns.values())

# Data Classification Registry
FRAUD_DATA_CATALOG = {
    'user_email': DataElement(
        name='user_email',
        classification=DataClassification.RESTRICTED,
        data_type='string',
        description='User email address for account identification',
        owner='Product Team',
        steward='Customer Data Steward',
        source_system='Authentication Service',
        created_date=datetime(2024, 1, 1),
        last_updated=datetime.now(),
        retention_period=timedelta(days=2555),  # 7 years
        processing_purpose='User authentication and fraud detection',
        legal_basis='Legitimate Interest - Fraud Prevention'
    ),
    
    'transaction_amount': DataElement(
        name='transaction_amount',
        classification=DataClassification.CONFIDENTIAL,
        data_type='decimal',
        description='Transaction amount in smallest currency unit',
        owner='Payments Team',
        steward='Transaction Data Steward',
        source_system='Payment Processor',
        created_date=datetime(2024, 1, 1),
        last_updated=datetime.now(),
        retention_period=timedelta(days=2555),  # 7 years
        processing_purpose='Fraud detection and compliance reporting',
        legal_basis='Contract Performance'
    ),
    
    'device_fingerprint': DataElement(
        name='device_fingerprint',
        classification=DataClassification.CONFIDENTIAL,
        data_type='string',
        description='Device fingerprint for fraud detection',
        owner='Security Team',
        steward='Fraud Detection Steward',
        source_system='Device Fingerprinting Service',
        created_date=datetime(2024, 1, 1),
        last_updated=datetime.now(),
        retention_period=timedelta(days=730),  # 2 years
        processing_purpose='Device recognition and fraud prevention',
        legal_basis='Legitimate Interest - Security'
    ),
    
    'fraud_score': DataElement(
        name='fraud_score',
        classification=DataClassification.CONFIDENTIAL,
        data_type='float',
        description='ML-generated fraud risk score (0-1)',
        owner='ML Team',
        steward='Fraud Detection Steward',
        source_system='ML Inference Service',
        created_date=datetime(2024, 1, 1),
        last_updated=datetime.now(),
        retention_period=timedelta(days=1825),  # 5 years
        processing_purpose='Fraud detection and model improvement',
        legal_basis='Legitimate Interest - Fraud Prevention'
    )
}
```

---

## 📈 Data Lineage and Cataloging

### **Data Lineage Tracking**
```python
from typing import Set, Optional
from dataclasses import dataclass, field
from datetime import datetime
import networkx as nx

@dataclass
class DataAsset:
    """Represents a data asset in the lineage graph."""
    id: str
    name: str
    asset_type: str  # table, view, file, api, model
    owner: str
    classification: DataClassification
    system: str
    schema_definition: Optional[Dict] = None
    tags: Set[str] = field(default_factory=set)
    
@dataclass 
class DataTransformation:
    """Represents a data transformation between assets."""
    id: str
    source_assets: List[str]
    target_asset: str
    transformation_type: str  # etl, ml_training, aggregation, join
    transformation_logic: str
    owner: str
    created_date: datetime
    business_rules: Optional[List[str]] = None

class DataLineageTracker:
    """Track and visualize data lineage across the platform."""
    
    def __init__(self):
        self.lineage_graph = nx.DiGraph()
        self.assets = {}
        self.transformations = {}
    
    def register_data_asset(self, asset: DataAsset) -> None:
        """Register a new data asset in the lineage graph."""
        self.assets[asset.id] = asset
        self.lineage_graph.add_node(
            asset.id,
            name=asset.name,
            asset_type=asset.asset_type,
            owner=asset.owner,
            classification=asset.classification.value,
            system=asset.system
        )
    
    def register_transformation(self, transformation: DataTransformation) -> None:
        """Register a data transformation in the lineage graph."""
        self.transformations[transformation.id] = transformation
        
        # Add edges from source assets to target asset
        for source_asset_id in transformation.source_assets:
            self.lineage_graph.add_edge(
                source_asset_id,
                transformation.target_asset,
                transformation_id=transformation.id,
                transformation_type=transformation.transformation_type,
                owner=transformation.owner
            )
    
    def get_upstream_assets(self, asset_id: str, max_depth: int = 10) -> Set[str]:
        """Get all upstream assets that feed into the given asset."""
        upstream_assets = set()
        
        def dfs_upstream(current_asset, current_depth):
            if current_depth >= max_depth:
                return
            
            predecessors = list(self.lineage_graph.predecessors(current_asset))
            upstream_assets.update(predecessors)
            
            for predecessor in predecessors:
                dfs_upstream(predecessor, current_depth + 1)
        
        dfs_upstream(asset_id, 0)
        return upstream_assets
    
    def get_downstream_assets(self, asset_id: str, max_depth: int = 10) -> Set[str]:
        """Get all downstream assets that consume the given asset."""
        downstream_assets = set()
        
        def dfs_downstream(current_asset, current_depth):
            if current_depth >= max_depth:
                return
            
            successors = list(self.lineage_graph.successors(current_asset))
            downstream_assets.update(successors)
            
            for successor in successors:
                dfs_downstream(successor, current_depth + 1)
        
        dfs_downstream(asset_id, 0)
        return downstream_assets
    
    def analyze_classification_flow(self, source_asset_id: str) -> Dict[str, List[str]]:
        """Analyze how data classification flows through lineage."""
        source_asset = self.assets[source_asset_id]
        source_classification = source_asset.classification
        
        downstream_assets = self.get_downstream_assets(source_asset_id)
        
        classification_violations = []
        classification_inheritance = []
        
        for asset_id in downstream_assets:
            asset = self.assets[asset_id]
            
            # Check if downstream asset has proper classification
            if asset.classification.value < source_classification.value:
                classification_violations.append({
                    'asset_id': asset_id,
                    'asset_name': asset.name,
                    'expected_classification': source_classification.value,
                    'actual_classification': asset.classification.value
                })
            else:
                classification_inheritance.append({
                    'asset_id': asset_id,
                    'asset_name': asset.name,
                    'inherited_classification': asset.classification.value
                })
        
        return {
            'violations': classification_violations,
            'proper_inheritance': classification_inheritance
        }
    
    def generate_impact_analysis(self, asset_id: str) -> Dict[str, any]:
        """Generate impact analysis for changes to a data asset."""
        upstream_assets = self.get_upstream_assets(asset_id)
        downstream_assets = self.get_downstream_assets(asset_id)
        
        # Analyze business impact
        critical_downstream = [
            asset_id for asset_id in downstream_assets
            if self.assets[asset_id].asset_type in ['api', 'dashboard', 'report']
        ]
        
        # Analyze security impact
        classification_levels = [
            self.assets[asset_id].classification
            for asset_id in downstream_assets
        ]
        max_classification = max(classification_levels) if classification_levels else None
        
        return {
            'asset_id': asset_id,
            'upstream_dependencies': len(upstream_assets),
            'downstream_consumers': len(downstream_assets),
            'critical_consumers': critical_downstream,
            'max_downstream_classification': max_classification.value if max_classification else None,
            'impact_score': self._calculate_impact_score(len(upstream_assets), len(downstream_assets), len(critical_downstream))
        }
    
    def _calculate_impact_score(self, upstream_count: int, downstream_count: int, critical_count: int) -> float:
        """Calculate impact score (0-100) for a data asset."""
        # Impact increases with more dependencies and critical consumers
        base_score = min((upstream_count + downstream_count) * 2, 70)
        critical_bonus = min(critical_count * 10, 30)
        return min(base_score + critical_bonus, 100)

# Initialize fraud detection data lineage
fraud_lineage = DataLineageTracker()

# Register data assets
fraud_lineage.register_data_asset(DataAsset(
    id="raw_transactions",
    name="Raw Transaction Data",
    asset_type="table",
    owner="Payments Team",
    classification=DataClassification.CONFIDENTIAL,
    system="PostgreSQL",
    schema_definition={
        "transaction_id": "UUID",
        "user_id": "UUID", 
        "amount": "DECIMAL(15,2)",
        "currency": "VARCHAR(3)",
        "timestamp": "TIMESTAMPTZ"
    }
))

fraud_lineage.register_data_asset(DataAsset(
    id="feature_engineered_transactions",
    name="Feature Engineered Transaction Data", 
    asset_type="view",
    owner="ML Team",
    classification=DataClassification.CONFIDENTIAL,
    system="Feature Store",
    schema_definition={
        "transaction_id": "UUID",
        "velocity_features": "JSONB",
        "behavioral_features": "JSONB", 
        "device_features": "JSONB"
    }
))

fraud_lineage.register_data_asset(DataAsset(
    id="fraud_predictions",
    name="Fraud Prediction Results",
    asset_type="table", 
    owner="ML Team",
    classification=DataClassification.CONFIDENTIAL,
    system="ML Inference Service",
    schema_definition={
        "transaction_id": "UUID",
        "fraud_score": "FLOAT",
        "risk_level": "VARCHAR(20)",
        "model_version": "VARCHAR(50)"
    }
))

# Register transformations
fraud_lineage.register_transformation(DataTransformation(
    id="feature_engineering_pipeline",
    source_assets=["raw_transactions"],
    target_asset="feature_engineered_transactions",
    transformation_type="etl",
    transformation_logic="Extract behavioral and velocity features from transaction history",
    owner="ML Engineering Team",
    created_date=datetime.now(),
    business_rules=[
        "Only include transactions from last 90 days for velocity features",
        "Exclude test transactions from feature calculation", 
        "Apply data quality filters before feature extraction"
    ]
))

fraud_lineage.register_transformation(DataTransformation(
    id="fraud_prediction_pipeline", 
    source_assets=["feature_engineered_transactions"],
    target_asset="fraud_predictions",
    transformation_type="ml_inference",
    transformation_logic="Apply ensemble ML model to generate fraud predictions",
    owner="ML Engineering Team", 
    created_date=datetime.now(),
    business_rules=[
        "Use latest approved model version",
        "Include model explainability features",
        "Log all predictions for audit trail"
    ]
))
```

---

## 🔒 Privacy Engineering

### **Privacy by Design Implementation**
```python
import hashlib
from cryptography.fernet import Fernet
from typing import Union, Optional, Dict, Any
import secrets

class PrivacyControls:
    """Implements privacy engineering controls."""
    
    def __init__(self):
        self.tokenization_key = self._load_tokenization_key()
        self.encryption_key = self._load_encryption_key()
        
    def tokenize_pii(self, pii_value: str, token_type: str = "reversible") -> str:
        """
        Tokenize PII data for privacy protection.
        
        Args:
            pii_value: The PII value to tokenize
            token_type: "reversible" or "irreversible"
            
        Returns:
            Tokenized value
        """
        if token_type == "reversible":
            # Format-preserving encryption
            return self._format_preserving_encrypt(pii_value)
        else:
            # One-way hash tokenization
            salt = secrets.token_bytes(32)
            hash_object = hashlib.pbkdf2_hmac('sha256', pii_value.encode(), salt, 100000)
            return f"token_{hash_object.hex()[:16]}"
    
    def detokenize_pii(self, token: str) -> Optional[str]:
        """
        Detokenize reversible tokens back to original PII.
        
        Args:
            token: The token to detokenize
            
        Returns:
            Original PII value or None if irreversible
        """
        if token.startswith("token_"):
            return None  # Irreversible token
        
        return self._format_preserving_decrypt(token)
    
    def pseudonymize_user_id(self, user_id: str, context: str) -> str:
        """
        Create pseudonymous user ID for specific context.
        
        Args:
            user_id: Original user ID
            context: Context for pseudonymization (e.g., "analytics", "ml_training")
            
        Returns:
            Pseudonymous ID unique to the context
        """
        combined = f"{user_id}:{context}:{self.tokenization_key}"
        return f"pseudo_{hashlib.sha256(combined.encode()).hexdigest()[:16]}"
    
    def anonymize_dataset(self, dataset: List[Dict[str, Any]], 
                         pii_fields: List[str],
                         quasi_identifiers: List[str],
                         k_value: int = 5) -> List[Dict[str, Any]]:
        """
        Apply k-anonymity to dataset by generalizing quasi-identifiers.
        
        Args:
            dataset: List of data records
            pii_fields: Fields to remove completely
            quasi_identifiers: Fields to generalize for k-anonymity
            k_value: Minimum group size for anonymity
            
        Returns:
            Anonymized dataset
        """
        anonymized_data = []
        
        for record in dataset:
            anonymized_record = record.copy()
            
            # Remove direct PII
            for pii_field in pii_fields:
                if pii_field in anonymized_record:
                    del anonymized_record[pii_field]
            
            # Generalize quasi-identifiers
            for qi_field in quasi_identifiers:
                if qi_field in anonymized_record:
                    anonymized_record[qi_field] = self._generalize_value(
                        anonymized_record[qi_field], qi_field
                    )
            
            anonymized_data.append(anonymized_record)
        
        # Apply k-anonymity grouping
        return self._apply_k_anonymity(anonymized_data, quasi_identifiers, k_value)
    
    def _generalize_value(self, value: Any, field_name: str) -> Any:
        """Generalize values for k-anonymity."""
        if field_name == 'age':
            # Generalize age to 5-year ranges
            age = int(value)
            range_start = (age // 5) * 5
            return f"{range_start}-{range_start + 4}"
        
        elif field_name == 'zip_code':
            # Generalize zip code to first 3 digits
            return str(value)[:3] + "**"
        
        elif field_name == 'salary':
            # Generalize salary to ranges
            salary = float(value)
            if salary < 50000:
                return "< 50k"
            elif salary < 100000:
                return "50k-100k"
            else:
                return "100k+"
        
        return value
    
    def differential_privacy_noise(self, true_value: float, 
                                 epsilon: float, 
                                 sensitivity: float) -> float:
        """
        Add Laplacian noise for differential privacy.
        
        Args:
            true_value: The true statistical value
            epsilon: Privacy budget parameter
            sensitivity: Global sensitivity of the query
            
        Returns:
            Noised value preserving differential privacy
        """
        import numpy as np
        
        # Laplacian mechanism
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale)
        
        return true_value + noise
    
    def generate_synthetic_data(self, original_dataset: List[Dict[str, Any]], 
                              num_synthetic_records: int,
                              privacy_budget: float = 1.0) -> List[Dict[str, Any]]:
        """
        Generate synthetic data preserving statistical properties while protecting privacy.
        
        Args:
            original_dataset: Original dataset to model
            num_synthetic_records: Number of synthetic records to generate
            privacy_budget: Differential privacy budget
            
        Returns:
            Synthetic dataset
        """
        # This is a simplified example - real implementation would use
        # more sophisticated techniques like GANs with differential privacy
        
        synthetic_data = []
        
        # Calculate statistics with differential privacy
        for i in range(num_synthetic_records):
            synthetic_record = {}
            
            for field in original_dataset[0].keys():
                if field in ['amount', 'transaction_count']:
                    # Numeric fields - use DP statistics
                    true_mean = np.mean([record[field] for record in original_dataset])
                    noisy_mean = self.differential_privacy_noise(
                        true_mean, privacy_budget, 1.0
                    )
                    # Generate synthetic value around noisy mean
                    synthetic_record[field] = max(0, np.random.normal(noisy_mean, noisy_mean * 0.1))
                
                elif field == 'user_segment':
                    # Categorical fields - sample with DP
                    segments = ['new', 'regular', 'vip']
                    synthetic_record[field] = np.random.choice(segments)
            
            synthetic_data.append(synthetic_record)
        
        return synthetic_data

class DataSubjectRightsHandler:
    """Handle GDPR data subject rights requests."""
    
    def __init__(self):
        self.privacy_controls = PrivacyControls()
        self.data_catalog = FRAUD_DATA_CATALOG
    
    async def handle_access_request(self, user_id: str) -> Dict[str, Any]:
        """
        Handle GDPR Article 15 - Right of Access request.
        
        Args:
            user_id: User requesting access to their data
            
        Returns:
            Complete export of user's personal data
        """
        user_data = {}
        
        # Collect data from all systems
        systems_to_query = [
            'user_profiles', 'transactions', 'fraud_assessments', 
            'device_fingerprints', 'audit_logs'
        ]
        
        for system in systems_to_query:
            try:
                system_data = await self._query_system_for_user_data(system, user_id)
                if system_data:
                    user_data[system] = system_data
            except Exception as e:
                user_data[system] = {'error': f'Could not retrieve data: {str(e)}'}
        
        # Include metadata about data processing
        user_data['processing_metadata'] = {
            'purposes': [
                'Fraud detection and prevention',
                'Transaction processing', 
                'Compliance and audit',
                'Service improvement'
            ],
            'legal_bases': [
                'Legitimate interest - fraud prevention',
                'Contract performance',
                'Legal obligation - AML/KYC'
            ],
            'retention_periods': {
                'transaction_data': '7 years',
                'fraud_assessments': '5 years', 
                'audit_logs': '10 years',
                'device_fingerprints': '2 years'
            },
            'data_recipients': [
                'Internal fraud detection team',
                'Compliance team',
                'External auditors (as required)'
            ]
        }
        
        return user_data
    
    async def handle_erasure_request(self, user_id: str, 
                                   reason: str = "withdrawal_of_consent") -> Dict[str, str]:
        """
        Handle GDPR Article 17 - Right to Erasure request.
        
        Args:
            user_id: User requesting data erasure
            reason: Reason for erasure request
            
        Returns:
            Status of erasure across all systems
        """
        erasure_results = {}
        
        # Check if erasure is legally possible
        legal_holds = await self._check_legal_holds(user_id)
        if legal_holds:
            return {
                'status': 'rejected',
                'reason': f'Data subject to legal holds: {", ".join(legal_holds)}',
                'partial_erasure': await self._perform_partial_erasure(user_id)
            }
        
        # Perform erasure across all systems
        systems_to_erase = [
            'user_profiles', 'transactions', 'fraud_assessments',
            'device_fingerprints', 'audit_logs', 'ml_training_data'
        ]
        
        for system in systems_to_erase:
            try:
                result = await self._erase_user_data_from_system(system, user_id)
                erasure_results[system] = result
            except Exception as e:
                erasure_results[system] = {'status': 'failed', 'error': str(e)}
        
        # Create erasure audit record
        await self._create_erasure_audit_record(user_id, reason, erasure_results)
        
        return erasure_results
    
    async def handle_portability_request(self, user_id: str) -> bytes:
        """
        Handle GDPR Article 20 - Right to Data Portability request.
        
        Args:
            user_id: User requesting data portability
            
        Returns:
            Structured data export in JSON format
        """
        portable_data = {}
        
        # Only include data provided by the user or generated through their use
        portable_systems = ['user_profiles', 'transactions', 'preferences']
        
        for system in portable_systems:
            system_data = await self._query_system_for_user_data(system, user_id)
            if system_data:
                # Structure data in machine-readable format
                portable_data[system] = self._structure_for_portability(system_data)
        
        # Create structured export
        export_data = {
            'user_id': user_id,
            'export_timestamp': datetime.now().isoformat(),
            'format_version': '1.0',
            'data': portable_data
        }
        
        return json.dumps(export_data, indent=2, default=str).encode('utf-8')
    
    async def _check_legal_holds(self, user_id: str) -> List[str]:
        """Check if user data is subject to legal holds."""
        holds = []
        
        # Check for active investigations
        if await self._has_active_fraud_investigation(user_id):
            holds.append("Active fraud investigation")
        
        # Check for regulatory requirements
        if await self._has_regulatory_retention_requirement(user_id):
            holds.append("Regulatory retention requirement")
        
        # Check for litigation hold
        if await self._has_litigation_hold(user_id):
            holds.append("Litigation hold")
        
        return holds
```

---

## 📊 Data Quality Management

### **Data Quality Framework**
```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import pandas as pd
import numpy as np

@dataclass
class DataQualityRule:
    """Represents a data quality rule."""
    name: str
    description: str
    rule_type: str  # completeness, accuracy, consistency, validity, timeliness, uniqueness
    severity: str   # critical, high, medium, low
    check_function: Callable
    threshold: Optional[float] = None
    enabled: bool = True

@dataclass
class DataQualityResult:
    """Result of data quality check."""
    rule_name: str
    passed: bool
    score: float  # 0.0 to 1.0
    details: Dict[str, Any]
    timestamp: datetime
    affected_records: Optional[List[str]] = None

class DataQualityEngine:
    """Comprehensive data quality monitoring and validation."""
    
    def __init__(self):
        self.rules = self._initialize_quality_rules()
        self.results_history = []
    
    def _initialize_quality_rules(self) -> List[DataQualityRule]:
        """Initialize standard data quality rules."""
        return [
            # Completeness Rules
            DataQualityRule(
                name="transaction_id_completeness",
                description="Transaction ID must not be null or empty",
                rule_type="completeness",
                severity="critical",
                check_function=self._check_not_null,
                threshold=0.999  # 99.9% completeness required
            ),
            
            DataQualityRule(
                name="user_id_completeness", 
                description="User ID must not be null or empty",
                rule_type="completeness",
                severity="critical",
                check_function=self._check_not_null,
                threshold=0.999
            ),
            
            # Validity Rules
            DataQualityRule(
                name="amount_validity",
                description="Transaction amount must be positive number",
                rule_type="validity", 
                severity="critical",
                check_function=self._check_positive_amount,
                threshold=0.995
            ),
            
            DataQualityRule(
                name="currency_validity",
                description="Currency must be valid ISO 4217 code",
                rule_type="validity",
                severity="high", 
                check_function=self._check_valid_currency,
                threshold=0.999
            ),
            
            DataQualityRule(
                name="email_validity",
                description="Email addresses must be valid format",
                rule_type="validity",
                severity="high",
                check_function=self._check_valid_email,
                threshold=0.99
            ),
            
            # Consistency Rules
            DataQualityRule(
                name="timestamp_consistency",
                description="Transaction timestamp must be within reasonable range",
                rule_type="consistency",
                severity="medium",
                check_function=self._check_timestamp_range,
                threshold=0.999
            ),
            
            # Uniqueness Rules
            DataQualityRule(
                name="transaction_id_uniqueness",
                description="Transaction IDs must be unique",
                rule_type="uniqueness", 
                severity="critical",
                check_function=self._check_uniqueness,
                threshold=1.0  # 100% uniqueness required
            ),
            
            # Accuracy Rules (business logic)
            DataQualityRule(
                name="fraud_score_range",
                description="Fraud scores must be between 0 and 1",
                rule_type="accuracy",
                severity="critical",
                check_function=self._check_fraud_score_range,
                threshold=1.0
            )
        ]
    
    def run_quality_checks(self, dataset: pd.DataFrame, 
                          rules_to_run: Optional[List[str]] = None) -> List[DataQualityResult]:
        """
        Run data quality checks on dataset.
        
        Args:
            dataset: DataFrame to validate
            rules_to_run: Specific rules to run, or None for all enabled rules
            
        Returns:
            List of data quality results
        """
        results = []
        
        active_rules = [rule for rule in self.rules if rule.enabled]
        if rules_to_run:
            active_rules = [rule for rule in active_rules if rule.name in rules_to_run]
        
        for rule in active_rules:
            try:
                result = self._execute_rule(rule, dataset)
                results.append(result)
                self.results_history.append(result)
            except Exception as e:
                results.append(DataQualityResult(
                    rule_name=rule.name,
                    passed=False,
                    score=0.0,
                    details={'error': str(e)},
                    timestamp=datetime.now()
                ))
        
        return results
    
    def _execute_rule(self, rule: DataQualityRule, dataset: pd.DataFrame) -> DataQualityResult:
        """Execute a single data quality rule."""
        start_time = datetime.now()
        
        # Execute the rule check function
        check_result = rule.check_function(dataset, rule)
        
        # Determine if rule passed based on threshold
        passed = check_result['score'] >= (rule.threshold or 1.0)
        
        return DataQualityResult(
            rule_name=rule.name,
            passed=passed,
            score=check_result['score'],
            details={
                **check_result,
                'execution_time_ms': (datetime.now() - start_time).total_seconds() * 1000
            },
            timestamp=start_time,
            affected_records=check_result.get('affected_records')
        )
    
    # Quality Check Functions
    def _check_not_null(self, dataset: pd.DataFrame, rule: DataQualityRule) -> Dict[str, Any]:
        """Check for null/empty values."""
        field_name = rule.name.split('_')[0]  # Extract field name from rule name
        
        if field_name not in dataset.columns:
            return {'score': 0.0, 'error': f'Field {field_name} not found'}
        
        total_rows = len(dataset)
        null_rows = dataset[field_name].isnull().sum()
        empty_rows = (dataset[field_name] == '').sum() if dataset[field_name].dtype == 'object' else 0
        
        invalid_rows = null_rows + empty_rows
        score = (total_rows - invalid_rows) / total_rows if total_rows > 0 else 1.0
        
        return {
            'score': score,
            'total_rows': total_rows,
            'null_rows': int(null_rows),
            'empty_rows': int(empty_rows),
            'valid_rows': total_rows - invalid_rows
        }
    
    def _check_positive_amount(self, dataset: pd.DataFrame, rule: DataQualityRule) -> Dict[str, Any]:
        """Check that amounts are positive."""
        if 'amount' not in dataset.columns:
            return {'score': 0.0, 'error': 'Amount field not found'}
        
        total_rows = len(dataset)
        valid_amounts = dataset['amount'] > 0
        valid_count = valid_amounts.sum()
        
        score = valid_count / total_rows if total_rows > 0 else 1.0
        
        invalid_records = dataset[~valid_amounts]['transaction_id'].tolist() if 'transaction_id' in dataset.columns else []
        
        return {
            'score': score,
            'total_rows': total_rows,
            'valid_amounts': int(valid_count),
            'invalid_amounts': total_rows - int(valid_count),
            'affected_records': invalid_records[:100]  # Limit to first 100
        }
    
    def _check_valid_currency(self, dataset: pd.DataFrame, rule: DataQualityRule) -> Dict[str, Any]:
        """Check for valid ISO 4217 currency codes."""
        valid_currencies = {'USD', 'EUR', 'GBP', 'CAD', 'AUD', 'JPY', 'CHF', 'CNY'}
        
        if 'currency' not in dataset.columns:
            return {'score': 0.0, 'error': 'Currency field not found'}
        
        total_rows = len(dataset)
        valid_currencies_mask = dataset['currency'].isin(valid_currencies)
        valid_count = valid_currencies_mask.sum()
        
        score = valid_count / total_rows if total_rows > 0 else 1.0
        
        invalid_currencies = dataset[~valid_currencies_mask]['currency'].value_counts().to_dict()
        
        return {
            'score': score,
            'total_rows': total_rows,
            'valid_count': int(valid_count),
            'invalid_currencies': invalid_currencies
        }
    
    def _check_valid_email(self, dataset: pd.DataFrame, rule: DataQualityRule) -> Dict[str, Any]:
        """Check for valid email format."""
        import re
        
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if 'email' not in dataset.columns:
            return {'score': 1.0, 'message': 'Email field not found - skipping'}
        
        total_rows = len(dataset)
        valid_emails = dataset['email'].apply(
            lambda x: bool(re.match(email_regex, str(x))) if pd.notna(x) else False
        )
        valid_count = valid_emails.sum()
        
        score = valid_count / total_rows if total_rows > 0 else 1.0
        
        return {
            'score': score,
            'total_rows': total_rows,
            'valid_emails': int(valid_count),
            'invalid_emails': total_rows - int(valid_count)
        }
    
    def _check_timestamp_range(self, dataset: pd.DataFrame, rule: DataQualityRule) -> Dict[str, Any]:
        """Check timestamps are within reasonable range."""
        if 'timestamp' not in dataset.columns:
            return {'score': 0.0, 'error': 'Timestamp field not found'}
        
        # Convert to datetime if needed
        timestamps = pd.to_datetime(dataset['timestamp'], errors='coerce')
        
        # Define reasonable range (not in future, not before 2020)
        now = datetime.now()
        min_date = datetime(2020, 1, 1)
        
        valid_timestamps = (timestamps >= min_date) & (timestamps <= now)
        valid_count = valid_timestamps.sum()
        total_rows = len(dataset)
        
        score = valid_count / total_rows if total_rows > 0 else 1.0
        
        return {
            'score': score,
            'total_rows': total_rows,
            'valid_timestamps': int(valid_count),
            'future_timestamps': int((timestamps > now).sum()),
            'past_timestamps': int((timestamps < min_date).sum())
        }
    
    def _check_uniqueness(self, dataset: pd.DataFrame, rule: DataQualityRule) -> Dict[str, Any]:
        """Check for unique values in specified field."""
        field_name = rule.name.split('_')[0]  # Extract field name from rule name
        
        if field_name not in dataset.columns:
            return {'score': 0.0, 'error': f'Field {field_name} not found'}
        
        total_rows = len(dataset)
        unique_values = dataset[field_name].nunique()
        
        score = unique_values / total_rows if total_rows > 0 else 1.0
        
        # Find duplicates
        duplicates = dataset[dataset.duplicated(subset=[field_name], keep=False)]
        duplicate_values = duplicates[field_name].value_counts().to_dict()
        
        return {
            'score': score,
            'total_rows': total_rows,
            'unique_values': unique_values,
            'duplicate_count': total_rows - unique_values,
            'duplicate_values': dict(list(duplicate_values.items())[:10])  # Top 10 duplicates
        }
    
    def _check_fraud_score_range(self, dataset: pd.DataFrame, rule: DataQualityRule) -> Dict[str, Any]:
        """Check fraud scores are in valid range [0, 1]."""
        if 'fraud_score' not in dataset.columns:
            return {'score': 1.0, 'message': 'Fraud score field not found - skipping'}
        
        total_rows = len(dataset)
        valid_range = (dataset['fraud_score'] >= 0) & (dataset['fraud_score'] <= 1)
        valid_count = valid_range.sum()
        
        score = valid_count / total_rows if total_rows > 0 else 1.0
        
        return {
            'score': score,
            'total_rows': total_rows,
            'valid_scores': int(valid_count),
            'out_of_range': total_rows - int(valid_count),
            'min_score': float(dataset['fraud_score'].min()),
            'max_score': float(dataset['fraud_score'].max())
        }
    
    def generate_quality_report(self, results: List[DataQualityResult]) -> Dict[str, Any]:
        """Generate comprehensive data quality report."""
        if not results:
            return {'status': 'no_results', 'message': 'No quality check results available'}
        
        # Overall statistics
        total_rules = len(results)
        passed_rules = sum(1 for r in results if r.passed)
        failed_rules = total_rules - passed_rules
        
        # Group by severity and rule type
        by_severity = {}
        by_type = {}
        
        for result in results:
            rule = next((r for r in self.rules if r.name == result.rule_name), None)
            if rule:
                severity = rule.severity
                rule_type = rule.rule_type
                
                if severity not in by_severity:
                    by_severity[severity] = {'passed': 0, 'failed': 0}
                if rule_type not in by_type:
                    by_type[rule_type] = {'passed': 0, 'failed': 0}
                
                status = 'passed' if result.passed else 'failed'
                by_severity[severity][status] += 1
                by_type[rule_type][status] += 1
        
        # Calculate overall quality score
        overall_score = sum(r.score for r in results) / len(results)
        
        # Identify critical failures
        critical_failures = [
            r for r in results
            if not r.passed and any(rule.severity == 'critical' for rule in self.rules if rule.name == r.rule_name)
        ]
        
        return {
            'overall_score': overall_score,
            'quality_grade': self._calculate_quality_grade(overall_score),
            'total_rules': total_rules,
            'passed_rules': passed_rules,
            'failed_rules': failed_rules,
            'critical_failures': len(critical_failures),
            'by_severity': by_severity,
            'by_rule_type': by_type,
            'detailed_results': [
                {
                    'rule_name': r.rule_name,
                    'passed': r.passed,
                    'score': r.score,
                    'severity': next((rule.severity for rule in self.rules if rule.name == r.rule_name), 'unknown')
                }
                for r in results
            ],
            'critical_failure_details': [
                {
                    'rule_name': r.rule_name,
                    'score': r.score,
                    'details': r.details
                }
                for r in critical_failures
            ],
            'report_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_quality_grade(self, score: float) -> str:
        """Calculate letter grade based on quality score."""
        if score >= 0.95:
            return 'A'
        elif score >= 0.90:
            return 'B'
        elif score >= 0.85:
            return 'C'
        elif score >= 0.75:
            return 'D'
        else:
            return 'F'
```

This comprehensive data governance framework provides enterprise-grade data management capabilities including automated classification, lineage tracking, privacy controls, and quality monitoring - all essential for regulatory compliance and data-driven decision making.

---

**Enterprise transformation is now complete!** 🎉