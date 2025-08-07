import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from django.utils import timezone
from django.db import transaction as db_transaction
from django.contrib.auth import get_user_model
from apps.transactions.models import Transaction
from .models import Alert, AlertRule, NotificationDelivery

User = get_user_model()
logger = logging.getLogger(__name__)


class AlertGenerator:
    """
    Service for generating alerts based on fraud detection results and configured rules.
    """
    
    def __init__(self):
        self.performance_metrics = {
            'alerts_generated': 0,
            'rules_evaluated': 0,
            'consolidations_performed': 0,
            'avg_generation_time_ms': 0.0
        }
    
    def evaluate_transaction(
        self, 
        transaction: Transaction, 
        fraud_result: Dict[str, Any], 
        user: User,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Alert]:
        """
        Evaluate a transaction against all active alert rules and generate alerts.
        
        Args:
            transaction: The transaction to evaluate
            fraud_result: Results from fraud detection analysis
            user: The user who owns the transaction
            context: Additional context data
            
        Returns:
            List of generated alerts
        """
        start_time = datetime.now()
        
        try:
            # Create alert context
            alert_context = self._create_alert_context(
                transaction, fraud_result, context
            )
            
            # Get applicable alert rules
            applicable_rules = self._get_applicable_rules(user, alert_context)
            
            # Evaluate rules and generate alerts
            generated_alerts = []
            for rule in applicable_rules:
                if self._evaluate_rule(rule, alert_context):
                    alert = self._create_alert(rule, transaction, alert_context, user)
                    if alert:
                        generated_alerts.append(alert)
            
            # Consolidate alerts if needed
            consolidated_alerts = self._consolidate_alerts(generated_alerts, user)
            
            # Update performance metrics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_metrics(len(consolidated_alerts), len(applicable_rules), processing_time)
            
            logger.info(
                f"Generated {len(consolidated_alerts)} alerts for transaction {transaction.transaction_id}"
            )
            
            return consolidated_alerts
            
        except Exception as e:
            logger.error(f"Alert generation failed for transaction {transaction.transaction_id}: {str(e)}")
            return []
    
    def _create_alert_context(
        self, 
        transaction: Transaction, 
        fraud_result: Dict[str, Any], 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create context object for alert rule evaluation."""
        
        alert_context = {
            # Transaction data
            'transaction_id': transaction.transaction_id,
            'user_id': transaction.user_id,
            'amount': float(transaction.amount),
            'currency': transaction.currency,
            'merchant_id': transaction.merchant_id,
            'payment_method': transaction.payment_method,
            'ip_address': transaction.ip_address,
            'timestamp': transaction.timestamp,
            
            # Fraud analysis results
            'fraud_score': fraud_result.get('fraud_probability', 0.0),
            'risk_score': fraud_result.get('risk_score', 0.0),
            'risk_level': fraud_result.get('risk_level', 'unknown'),
            'confidence': fraud_result.get('confidence', 0.0),
            'ml_results': fraud_result.get('ml_results', {}),
            'rule_results': fraud_result.get('rule_results', []),
            
            # Additional context
            'context_data': context or {},
            'evaluation_timestamp': timezone.now()
        }
        
        # Add derived features
        alert_context.update(self._calculate_derived_features(transaction, fraud_result))
        
        return alert_context
    
    def _calculate_derived_features(
        self, 
        transaction: Transaction, 
        fraud_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate derived features for rule evaluation."""
        
        features = {}
        
        # Time-based features
        now = timezone.now()
        transaction_time = transaction.timestamp
        
        features['is_weekend'] = transaction_time.weekday() >= 5
        features['is_night_time'] = transaction_time.hour < 6 or transaction_time.hour > 22
        features['hour_of_day'] = transaction_time.hour
        features['day_of_week'] = transaction_time.weekday()
        
        # Amount-based features
        amount = float(transaction.amount)
        features['is_high_amount'] = amount > 1000
        features['is_round_amount'] = amount == int(amount)
        features['amount_category'] = self._categorize_amount(amount)
        
        # Fraud score features
        fraud_score = fraud_result.get('fraud_probability', 0.0)
        features['fraud_score_category'] = self._categorize_fraud_score(fraud_score)
        features['is_high_risk'] = fraud_score > 0.8
        features['is_medium_risk'] = 0.5 <= fraud_score <= 0.8
        
        # ML model features
        ml_results = fraud_result.get('ml_results', {})
        if ml_results:
            features['ml_confidence'] = ml_results.get('ml_confidence', 0.0)
            features['model_agreement'] = self._calculate_model_agreement(ml_results)
        
        return features
    
    def _categorize_amount(self, amount: float) -> str:
        """Categorize transaction amount."""
        if amount < 10:
            return 'micro'
        elif amount < 100:
            return 'small'
        elif amount < 1000:
            return 'medium'
        elif amount < 10000:
            return 'large'
        else:
            return 'very_large'
    
    def _categorize_fraud_score(self, score: float) -> str:
        """Categorize fraud score."""
        if score < 0.2:
            return 'very_low'
        elif score < 0.4:
            return 'low'
        elif score < 0.6:
            return 'medium'
        elif score < 0.8:
            return 'high'
        else:
            return 'very_high'
    
    def _calculate_model_agreement(self, ml_results: Dict[str, Any]) -> float:
        """Calculate agreement between ML models."""
        model_predictions = ml_results.get('model_predictions', {})
        if len(model_predictions) < 2:
            return 1.0
        
        scores = [pred.get('fraud_probability', 0.0) for pred in model_predictions.values()]
        if not scores:
            return 1.0
        
        # Calculate coefficient of variation (lower = more agreement)
        mean_score = sum(scores) / len(scores)
        if mean_score == 0:
            return 1.0
        
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        std_dev = variance ** 0.5
        cv = std_dev / mean_score
        
        # Convert to agreement score (0-1, higher = more agreement)
        return max(0.0, 1.0 - cv)
    
    def _get_applicable_rules(self, user: User, alert_context: Dict[str, Any]) -> List[AlertRule]:
        """Get alert rules that are applicable to the current context."""
        
        # Get all active rules for the user
        rules = AlertRule.objects.filter(
            owner=user,
            is_active=True
        ).order_by('priority', 'name')
        
        applicable_rules = []
        
        for rule in rules:
            # Check basic thresholds
            if rule.fraud_score_threshold is not None:
                if alert_context['fraud_score'] < rule.fraud_score_threshold:
                    continue
            
            if rule.amount_threshold is not None:
                if alert_context['amount'] < float(rule.amount_threshold):
                    continue
            
            applicable_rules.append(rule)
        
        return applicable_rules
    
    def _evaluate_rule(self, rule: AlertRule, alert_context: Dict[str, Any]) -> bool:
        """Evaluate a single alert rule against the context."""
        
        try:
            conditions = rule.conditions
            if not isinstance(conditions, dict):
                return False
            
            # Get meta configuration
            meta = conditions.get('_meta', {})
            required_conditions = meta.get('required_conditions', 1)
            
            # Evaluate each condition
            met_conditions = 0
            
            for condition_name, condition_config in conditions.items():
                if condition_name.startswith('_'):  # Skip meta fields
                    continue
                
                if self._evaluate_condition(condition_config, alert_context):
                    met_conditions += 1
            
            # Check if enough conditions are met
            return met_conditions >= required_conditions
            
        except Exception as e:
            logger.error(f"Rule evaluation failed for rule {rule.name}: {str(e)}")
            return False
    
    def _evaluate_condition(
        self, 
        condition_config: Dict[str, Any], 
        alert_context: Dict[str, Any]
    ) -> bool:
        """Evaluate a single condition."""
        
        condition_type = condition_config.get('type', 'threshold')
        field = condition_config.get('field', '')
        
        # Get value from context
        value = alert_context.get(field, 0)
        
        if condition_type == 'threshold':
            return self._evaluate_threshold_condition(condition_config, value)
        elif condition_type == 'categorical':
            return self._evaluate_categorical_condition(condition_config, value)
        elif condition_type == 'range':
            return self._evaluate_range_condition(condition_config, value)
        elif condition_type == 'time_based':
            return self._evaluate_time_condition(condition_config, alert_context)
        else:
            logger.warning(f"Unknown condition type: {condition_type}")
            return False
    
    def _evaluate_threshold_condition(
        self, 
        condition_config: Dict[str, Any], 
        value: Any
    ) -> bool:
        """Evaluate threshold-based condition."""
        
        threshold = condition_config.get('threshold', 0)
        operator = condition_config.get('operator', '>')
        
        try:
            numeric_value = float(value)
            numeric_threshold = float(threshold)
            
            if operator == '>':
                return numeric_value > numeric_threshold
            elif operator == '>=':
                return numeric_value >= numeric_threshold
            elif operator == '<':
                return numeric_value < numeric_threshold
            elif operator == '<=':
                return numeric_value <= numeric_threshold
            elif operator == '==':
                return numeric_value == numeric_threshold
            elif operator == '!=':
                return numeric_value != numeric_threshold
            else:
                return False
                
        except (ValueError, TypeError):
            return False
    
    def _evaluate_categorical_condition(
        self, 
        condition_config: Dict[str, Any], 
        value: Any
    ) -> bool:
        """Evaluate categorical condition."""
        
        allowed_values = condition_config.get('values', [])
        return str(value) in [str(v) for v in allowed_values]
    
    def _evaluate_range_condition(
        self, 
        condition_config: Dict[str, Any], 
        value: Any
    ) -> bool:
        """Evaluate range-based condition."""
        
        min_value = condition_config.get('min')
        max_value = condition_config.get('max')
        
        try:
            numeric_value = float(value)
            
            if min_value is not None and numeric_value < float(min_value):
                return False
            if max_value is not None and numeric_value > float(max_value):
                return False
            
            return True
            
        except (ValueError, TypeError):
            return False
    
    def _evaluate_time_condition(
        self, 
        condition_config: Dict[str, Any], 
        alert_context: Dict[str, Any]
    ) -> bool:
        """Evaluate time-based condition."""
        
        condition_subtype = condition_config.get('subtype', 'hour_range')
        
        if condition_subtype == 'hour_range':
            start_hour = condition_config.get('start_hour', 0)
            end_hour = condition_config.get('end_hour', 23)
            current_hour = alert_context.get('hour_of_day', 0)
            
            if start_hour <= end_hour:
                return start_hour <= current_hour <= end_hour
            else:  # Overnight range
                return current_hour >= start_hour or current_hour <= end_hour
        
        elif condition_subtype == 'day_of_week':
            allowed_days = condition_config.get('days', [])
            current_day = alert_context.get('day_of_week', 0)
            return current_day in allowed_days
        
        elif condition_subtype == 'weekend':
            is_weekend = alert_context.get('is_weekend', False)
            return is_weekend == condition_config.get('value', True)
        
        return False  
  
    def _create_alert(
        self, 
        rule: AlertRule, 
        transaction: Transaction, 
        alert_context: Dict[str, Any], 
        user: User
    ) -> Optional[Alert]:
        """Create an alert based on a triggered rule."""
        
        try:
            with db_transaction.atomic():
                # Generate alert title and message
                title = self._generate_alert_title(rule, alert_context)
                message = self._generate_alert_message(rule, alert_context)
                
                # Extract risk factors
                risk_factors = self._extract_risk_factors(alert_context)
                
                # Create the alert
                alert = Alert.objects.create(
                    alert_type=rule.alert_type,
                    severity=rule.severity,
                    transaction=transaction,
                    fraud_score=alert_context['fraud_score'],
                    risk_factors=risk_factors,
                    title=title,
                    message=message,
                    context_data=alert_context.get('context_data', {}),
                    rule_triggered=rule,
                    owner=user
                )
                
                logger.info(f"Created alert {alert.id} for transaction {transaction.transaction_id}")
                return alert
                
        except Exception as e:
            logger.error(f"Failed to create alert for rule {rule.name}: {str(e)}")
            return None
    
    def _generate_alert_title(self, rule: AlertRule, alert_context: Dict[str, Any]) -> str:
        """Generate alert title based on rule and context."""
        
        base_title = f"{rule.alert_type.replace('_', ' ').title()} Alert"
        
        # Add context-specific information
        if alert_context.get('fraud_score', 0) > 0.8:
            base_title = f"HIGH RISK: {base_title}"
        elif alert_context.get('amount', 0) > 10000:
            base_title = f"LARGE AMOUNT: {base_title}"
        
        return base_title
    
    def _generate_alert_message(self, rule: AlertRule, alert_context: Dict[str, Any]) -> str:
        """Generate alert message based on rule and context."""
        
        message_parts = [
            f"Alert triggered by rule: {rule.name}",
            f"Transaction ID: {alert_context['transaction_id']}",
            f"Amount: {alert_context['amount']} {alert_context['currency']}",
            f"Fraud Score: {alert_context['fraud_score']:.2%}",
            f"Risk Level: {alert_context['risk_level']}"
        ]
        
        # Add specific risk factors
        if alert_context.get('is_night_time'):
            message_parts.append("⚠️ Transaction occurred during night hours")
        
        if alert_context.get('is_high_amount'):
            message_parts.append("⚠️ High transaction amount")
        
        if alert_context.get('fraud_score', 0) > 0.8:
            message_parts.append("🚨 Very high fraud probability detected")
        
        return "\n".join(message_parts)
    
    def _extract_risk_factors(self, alert_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key risk factors from alert context."""
        
        risk_factors = {}
        
        # Fraud score factors
        fraud_score = alert_context.get('fraud_score', 0)
        if fraud_score > 0.8:
            risk_factors['high_fraud_score'] = fraud_score
        
        # Amount factors
        amount = alert_context.get('amount', 0)
        if amount > 1000:
            risk_factors['high_amount'] = amount
        
        # Time factors
        if alert_context.get('is_night_time'):
            risk_factors['night_transaction'] = True
        
        if alert_context.get('is_weekend'):
            risk_factors['weekend_transaction'] = True
        
        # ML model factors
        ml_results = alert_context.get('ml_results', {})
        if ml_results.get('ml_confidence', 0) > 0.8:
            risk_factors['high_ml_confidence'] = ml_results['ml_confidence']
        
        # Rule-based factors
        rule_results = alert_context.get('rule_results', [])
        triggered_rules = [r for r in rule_results if r.get('triggered')]
        if len(triggered_rules) > 1:
            risk_factors['multiple_rules_triggered'] = len(triggered_rules)
        
        return risk_factors
    
    def _consolidate_alerts(self, alerts: List[Alert], user: User) -> List[Alert]:
        """Consolidate similar alerts within time windows."""
        
        if not alerts:
            return alerts
        
        consolidated = []
        processed_alerts = set()
        
        for alert in alerts:
            if alert.id in processed_alerts:
                continue
            
            # Find similar alerts within consolidation window
            similar_alerts = self._find_similar_alerts(alert, alerts, user)
            
            if len(similar_alerts) > 1:
                # Consolidate similar alerts
                consolidated_alert = self._merge_alerts(similar_alerts)
                if consolidated_alert:
                    consolidated.append(consolidated_alert)
                    processed_alerts.update(a.id for a in similar_alerts)
                    self.performance_metrics['consolidations_performed'] += 1
            else:
                consolidated.append(alert)
                processed_alerts.add(alert.id)
        
        return consolidated
    
    def _find_similar_alerts(
        self, 
        target_alert: Alert, 
        all_alerts: List[Alert], 
        user: User
    ) -> List[Alert]:
        """Find alerts similar to the target alert."""
        
        similar = [target_alert]
        consolidation_window = timedelta(minutes=10)  # Default window
        
        # Get consolidation window from rule if available
        if target_alert.rule_triggered:
            consolidation_window = target_alert.rule_triggered.consolidation_window
        
        cutoff_time = target_alert.created_at - consolidation_window
        
        # Check recent alerts of the same type
        recent_alerts = Alert.objects.filter(
            owner=user,
            alert_type=target_alert.alert_type,
            created_at__gte=cutoff_time,
            status='pending'
        ).exclude(id=target_alert.id)
        
        for alert in recent_alerts:
            if self._are_alerts_similar(target_alert, alert):
                similar.append(alert)
        
        return similar
    
    def _are_alerts_similar(self, alert1: Alert, alert2: Alert) -> bool:
        """Check if two alerts are similar enough to consolidate."""
        
        # Same alert type
        if alert1.alert_type != alert2.alert_type:
            return False
        
        # Same user
        if alert1.transaction.user_id != alert2.transaction.user_id:
            return False
        
        # Similar fraud scores (within 0.1)
        score_diff = abs(alert1.fraud_score - alert2.fraud_score)
        if score_diff > 0.1:
            return False
        
        # Same severity level
        if alert1.severity != alert2.severity:
            return False
        
        return True
    
    def _merge_alerts(self, alerts: List[Alert]) -> Optional[Alert]:
        """Merge multiple similar alerts into one consolidated alert."""
        
        if not alerts:
            return None
        
        try:
            # Use the first alert as the base
            primary_alert = alerts[0]
            
            # Update the primary alert with consolidated information
            transaction_ids = [a.transaction.transaction_id for a in alerts]
            consolidated_message = f"Consolidated alert for {len(alerts)} similar transactions:\n"
            consolidated_message += f"Transaction IDs: {', '.join(transaction_ids)}\n"
            consolidated_message += f"Average Fraud Score: {sum(a.fraud_score for a in alerts) / len(alerts):.2%}"
            
            primary_alert.message = consolidated_message
            primary_alert.title = f"CONSOLIDATED: {primary_alert.title}"
            
            # Merge risk factors
            merged_risk_factors = {}
            for alert in alerts:
                merged_risk_factors.update(alert.risk_factors)
            
            merged_risk_factors['consolidated_count'] = len(alerts)
            merged_risk_factors['transaction_ids'] = transaction_ids
            
            primary_alert.risk_factors = merged_risk_factors
            primary_alert.save()
            
            # Delete the other alerts
            for alert in alerts[1:]:
                alert.delete()
            
            return primary_alert
            
        except Exception as e:
            logger.error(f"Failed to merge alerts: {str(e)}")
            return alerts[0] if alerts else None
    
    def _update_metrics(self, alerts_generated: int, rules_evaluated: int, processing_time: float):
        """Update performance metrics."""
        
        self.performance_metrics['alerts_generated'] += alerts_generated
        self.performance_metrics['rules_evaluated'] += rules_evaluated
        
        # Update average processing time (exponential moving average)
        alpha = 0.1
        current_avg = self.performance_metrics['avg_generation_time_ms']
        self.performance_metrics['avg_generation_time_ms'] = (
            alpha * processing_time + (1 - alpha) * current_avg
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()
    
    def reset_metrics(self):
        """Reset performance metrics."""
        self.performance_metrics = {
            'alerts_generated': 0,
            'rules_evaluated': 0,
            'consolidations_performed': 0,
            'avg_generation_time_ms': 0.0
        }


# Global instance for use across the application
alert_generator = AlertGenerator()