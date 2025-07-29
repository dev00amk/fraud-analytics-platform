import numpy as np
from django.conf import settings
from django.core.cache import cache
from .models import FraudRule
import logging

logger = logging.getLogger(__name__)


class FraudDetectionService:
    """Core fraud detection service."""
    
    def __init__(self):
        self.low_threshold = settings.FRAUD_THRESHOLD_LOW
        self.medium_threshold = settings.FRAUD_THRESHOLD_MEDIUM
        self.high_threshold = settings.FRAUD_THRESHOLD_HIGH
    
    def analyze_transaction(self, transaction_data, user):
        """Analyze a transaction for fraud indicators."""
        try:
            # Calculate fraud score
            fraud_score = self._calculate_fraud_score(transaction_data, user)
            
            # Determine risk level
            risk_level = self._determine_risk_level(fraud_score)
            
            # Get recommendation
            recommendation = self._get_recommendation(risk_level, fraud_score)
            
            # Apply rules
            rule_results = self._apply_rules(transaction_data, user)
            
            return {
                'fraud_score': fraud_score,
                'risk_level': risk_level,
                'recommendation': recommendation,
                'rule_results': rule_results,
                'analysis_timestamp': transaction_data.get('timestamp')
            }
        
        except Exception as e:
            logger.error(f"Fraud analysis error: {str(e)}")
            return {
                'fraud_score': 0,
                'risk_level': 'unknown',
                'recommendation': 'manual_review',
                'error': 'Analysis failed'
            }
    
    def _calculate_fraud_score(self, transaction_data, user):
        """Calculate fraud score using various factors."""
        score = 0
        
        # Amount-based scoring
        amount = float(transaction_data.get('amount', 0))
        if amount > 1000:
            score += 20
        elif amount > 500:
            score += 10
        
        # Velocity checks (simplified)
        user_id = transaction_data.get('user_id')
        recent_transactions = self._get_recent_transaction_count(user_id)
        if recent_transactions > 5:
            score += 30
        elif recent_transactions > 3:
            score += 15
        
        # Geographic risk (simplified)
        ip_address = transaction_data.get('ip_address')
        if self._is_high_risk_ip(ip_address):
            score += 25
        
        # Time-based risk
        if self._is_unusual_time(transaction_data.get('timestamp')):
            score += 10
        
        return min(score, 100)  # Cap at 100
    
    def _determine_risk_level(self, fraud_score):
        """Determine risk level based on fraud score."""
        if fraud_score >= self.high_threshold:
            return 'high'
        elif fraud_score >= self.medium_threshold:
            return 'medium'
        elif fraud_score >= self.low_threshold:
            return 'low'
        else:
            return 'very_low'
    
    def _get_recommendation(self, risk_level, fraud_score):
        """Get recommendation based on risk level."""
        recommendations = {
            'very_low': 'approve',
            'low': 'approve',
            'medium': 'manual_review',
            'high': 'decline'
        }
        return recommendations.get(risk_level, 'manual_review')
    
    def _apply_rules(self, transaction_data, user):
        """Apply fraud detection rules."""
        rules = FraudRule.objects.filter(is_active=True, owner=user)
        results = []
        
        for rule in rules:
            result = self._evaluate_rule(rule, transaction_data)
            results.append({
                'rule_name': rule.name,
                'triggered': result,
                'action': rule.action if result else None
            })
        
        return results
    
    def _evaluate_rule(self, rule, transaction_data):
        """Evaluate a single fraud rule."""
        # Simplified rule evaluation
        # In production, this would be more sophisticated
        return False
    
    def _get_recent_transaction_count(self, user_id):
        """Get count of recent transactions for user."""
        cache_key = f"user_transactions_{user_id}"
        count = cache.get(cache_key, 0)
        return count
    
    def _is_high_risk_ip(self, ip_address):
        """Check if IP address is high risk."""
        # Simplified check - in production, use IP reputation services
        return False
    
    def _is_unusual_time(self, timestamp):
        """Check if transaction time is unusual."""
        # Simplified check - in production, analyze user patterns
        return False