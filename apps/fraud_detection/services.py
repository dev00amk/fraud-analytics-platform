import numpy as np
import asyncio
from typing import Dict, List, Optional, Any
from django.conf import settings
from django.core.cache import cache
from .models import FraudRule
from apps.ml_models.ensemble_model import EnsembleInferenceEngine
from apps.ml_models.feature_engineering import FeatureEngineeringPipeline
import logging
import redis
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class AdvancedFraudDetectionService:
    """
    Advanced fraud detection service with ML models and real-time feature engineering.
    Integrates ensemble models (XGBoost + LSTM + GNN + Transformer) for production-grade fraud detection.
    """
    
    def __init__(self):
        self.low_threshold = settings.FRAUD_THRESHOLD_LOW
        self.medium_threshold = settings.FRAUD_THRESHOLD_MEDIUM
        self.high_threshold = settings.FRAUD_THRESHOLD_HIGH
        
        # Initialize Redis client for feature engineering
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST if hasattr(settings, 'REDIS_HOST') else 'localhost',
                port=settings.REDIS_PORT if hasattr(settings, 'REDIS_PORT') else 6379,
                db=settings.REDIS_DB if hasattr(settings, 'REDIS_DB') else 0,
                decode_responses=True
            )
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
        
        # Initialize feature engineering pipeline
        self.feature_pipeline = FeatureEngineeringPipeline(
            redis_client=self.redis_client,
            enable_caching=True,
            cache_ttl=300,  # 5 minutes
            max_workers=4
        )
        
        # Initialize ensemble model
        try:
            self.ensemble_engine = EnsembleInferenceEngine(
                config_path=getattr(settings, 'ML_MODEL_CONFIG_PATH', 'ml_models/config.json')
            )
        except Exception as e:
            logger.warning(f"Ensemble model initialization failed: {e}")
            self.ensemble_engine = None
        
        # Performance tracking
        self.performance_metrics = {
            'total_predictions': 0,
            'avg_inference_time': 0.0,
            'model_accuracy': 0.85,
            'false_positive_rate': 0.05,
        }
    
    async def analyze_transaction_async(
        self,
        transaction_data: Dict[str, Any],
        user: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Asynchronous transaction analysis with advanced ML models.
        
        Args:
            transaction_data: Transaction to analyze
            user: User object for context
            context: Additional context (user history, merchant data, etc.)
            
        Returns:
            Comprehensive fraud analysis results
        """
        start_time = datetime.now()
        
        try:
            # Extract comprehensive features
            feature_extraction_start = datetime.now()
            features = await self._extract_features_async(transaction_data, context)
            feature_time = (datetime.now() - feature_extraction_start).total_seconds() * 1000
            
            # Run ensemble model prediction
            ml_prediction_start = datetime.now()
            ml_results = await self._run_ml_prediction_async(
                transaction_data, features, context
            )
            ml_time = (datetime.now() - ml_prediction_start).total_seconds() * 1000
            
            # Apply business rules
            rules_start = datetime.now()
            rule_results = await self._apply_rules_async(transaction_data, user, features)
            rules_time = (datetime.now() - rules_start).total_seconds() * 1000
            
            # Combine results
            final_results = self._combine_analysis_results(
                ml_results, rule_results, features, transaction_data
            )
            
            # Calculate total processing time
            total_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Add performance metrics
            final_results.update({
                'performance_metrics': {
                    'total_time_ms': total_time,
                    'feature_extraction_time_ms': feature_time,
                    'ml_prediction_time_ms': ml_time,
                    'rules_evaluation_time_ms': rules_time,
                    'features_extracted': len(features.to_dict()) if features else 0,
                },
                'analysis_timestamp': datetime.now().isoformat(),
                'model_version': '2.0.0',
                'service_version': 'advanced'
            })
            
            # Update performance tracking
            self._update_performance_metrics(total_time, final_results)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Advanced fraud analysis error: {str(e)}")
            return self._get_fallback_result(transaction_data, str(e))
    
    def analyze_transaction(
        self,
        transaction_data: Dict[str, Any],
        user: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for transaction analysis.
        """
        try:
            # Run async analysis in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.analyze_transaction_async(transaction_data, user, context)
            )
            loop.close()
            return result
        except Exception as e:
            logger.error(f"Sync fraud analysis error: {str(e)}")
            return self._get_fallback_result(transaction_data, str(e))
    
    async def _extract_features_async(
        self,
        transaction_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ):
        """Extract comprehensive features using the feature engineering pipeline."""
        try:
            # Get historical data from context
            user_transactions = context.get('user_transactions', []) if context else []
            merchant_transactions = context.get('merchant_transactions', []) if context else []
            
            # Extract features
            features = self.feature_pipeline.extract_features(
                transaction=transaction_data,
                user_transactions=user_transactions,
                merchant_transactions=merchant_transactions,
                use_cache=True
            )
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None
    
    async def _run_ml_prediction_async(
        self,
        transaction_data: Dict[str, Any],
        features,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run ML model predictions using the ensemble."""
        if not self.ensemble_engine or not features:
            return self._get_basic_ml_prediction(transaction_data)
        
        try:
            # Prepare data for ensemble
            user_transactions = context.get('user_transactions', {}) if context else {}
            transaction_graph = context.get('transaction_graph', []) if context else []
            
            # Run ensemble prediction
            ensemble_result = self.ensemble_engine.predict(
                transaction=transaction_data,
                user_transactions=user_transactions,
                transaction_graph=transaction_graph,
                use_cache=True
            )
            
            return {
                'ml_fraud_probability': ensemble_result.fraud_probability,
                'ml_risk_score': ensemble_result.risk_score,
                'ml_confidence': ensemble_result.confidence,
                'model_predictions': ensemble_result.model_predictions,
                'ensemble_weights': ensemble_result.ensemble_weights,
                'ml_inference_time_ms': ensemble_result.inference_time_ms,
                'model_type': 'ensemble',
                'models_used': list(ensemble_result.model_predictions.keys())
            }
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return self._get_basic_ml_prediction(transaction_data)
    
    async def _apply_rules_async(
        self,
        transaction_data: Dict[str, Any],
        user: Any,
        features
    ) -> List[Dict[str, Any]]:
        """Apply fraud detection rules with feature-based evaluation."""
        try:
            rules = FraudRule.objects.filter(is_active=True, owner=user)
            results = []
            
            for rule in rules:
                result = await self._evaluate_rule_async(rule, transaction_data, features)
                results.append({
                    'rule_id': str(rule.id),
                    'rule_name': rule.name,
                    'triggered': result['triggered'],
                    'confidence': result['confidence'],
                    'action': rule.action if result['triggered'] else None,
                    'conditions_met': result['conditions_met'],
                    'evaluation_time_ms': result['evaluation_time_ms']
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Rules evaluation failed: {e}")
            return []
    
    async def _evaluate_rule_async(
        self,
        rule: FraudRule,
        transaction_data: Dict[str, Any],
        features
    ) -> Dict[str, Any]:
        """Evaluate a single fraud rule with advanced conditions."""
        start_time = datetime.now()
        
        try:
            conditions = rule.conditions
            if not isinstance(conditions, dict):
                return {
                    'triggered': False,
                    'confidence': 0.0,
                    'conditions_met': [],
                    'evaluation_time_ms': 0.0
                }
            
            feature_dict = features.to_dict() if features else {}
            conditions_met = []
            
            # Evaluate each condition
            for condition_name, condition_config in conditions.items():
                condition_result = self._evaluate_condition(
                    condition_config, transaction_data, feature_dict
                )
                
                if condition_result['met']:
                    conditions_met.append({
                        'condition': condition_name,
                        'value': condition_result['value'],
                        'threshold': condition_result['threshold'],
                        'confidence': condition_result['confidence']
                    })
            
            # Determine if rule is triggered
            required_conditions = conditions.get('_meta', {}).get('required_conditions', 1)
            triggered = len(conditions_met) >= required_conditions
            
            # Calculate overall confidence
            if conditions_met:
                confidence = np.mean([c['confidence'] for c in conditions_met])
            else:
                confidence = 0.0
            
            evaluation_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                'triggered': triggered,
                'confidence': confidence,
                'conditions_met': conditions_met,
                'evaluation_time_ms': evaluation_time
            }
            
        except Exception as e:
            logger.error(f"Rule evaluation failed: {e}")
            return {
                'triggered': False,
                'confidence': 0.0,
                'conditions_met': [],
                'evaluation_time_ms': 0.0
            }
    
    def _evaluate_condition(
        self,
        condition_config: Dict[str, Any],
        transaction_data: Dict[str, Any],
        feature_dict: Dict[str, float]
    ) -> Dict[str, Any]:
        """Evaluate a single condition."""
        condition_type = condition_config.get('type', 'threshold')
        field = condition_config.get('field', '')
        
        # Get value from transaction data or features
        if field.startswith('feature_'):
            value = feature_dict.get(field, 0.0)
        else:
            value = transaction_data.get(field, 0.0)
        
        if condition_type == 'threshold':
            threshold = condition_config.get('threshold', 0.0)
            operator = condition_config.get('operator', '>')
            
            if operator == '>':
                met = float(value) > threshold
            elif operator == '>=':
                met = float(value) >= threshold
            elif operator == '<':
                met = float(value) < threshold
            elif operator == '<=':
                met = float(value) <= threshold
            elif operator == '==':
                met = float(value) == threshold
            else:
                met = False
            
            # Calculate confidence based on how far from threshold
            if met:
                distance = abs(float(value) - threshold)
                confidence = min(distance / max(threshold, 1.0), 1.0)
            else:
                confidence = 0.0
            
            return {
                'met': met,
                'value': float(value),
                'threshold': threshold,
                'confidence': confidence
            }
        
        elif condition_type == 'categorical':
            allowed_values = condition_config.get('values', [])
            met = str(value) in allowed_values
            
            return {
                'met': met,
                'value': str(value),
                'threshold': allowed_values,
                'confidence': 1.0 if met else 0.0
            }
        
        else:
            return {
                'met': False,
                'value': value,
                'threshold': None,
                'confidence': 0.0
            }
    
    def _combine_analysis_results(
        self,
        ml_results: Dict[str, Any],
        rule_results: List[Dict[str, Any]],
        features,
        transaction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine ML predictions and rule results into final analysis."""
        
        # Get ML predictions
        ml_fraud_prob = ml_results.get('ml_fraud_probability', 0.5)
        ml_risk_score = ml_results.get('ml_risk_score', 0.5)
        ml_confidence = ml_results.get('ml_confidence', 0.5)
        
        # Analyze rule results
        triggered_rules = [r for r in rule_results if r['triggered']]
        high_confidence_rules = [r for r in triggered_rules if r['confidence'] > 0.8]
        
        # Calculate rule-based risk
        if triggered_rules:
            rule_risk_score = min(len(triggered_rules) * 0.2, 1.0)
            rule_confidence = np.mean([r['confidence'] for r in triggered_rules])
        else:
            rule_risk_score = 0.0
            rule_confidence = 0.0
        
        # Combine ML and rule-based scores
        ml_weight = 0.7
        rule_weight = 0.3
        
        combined_fraud_prob = (ml_weight * ml_fraud_prob + rule_weight * rule_risk_score)
        combined_risk_score = (ml_weight * ml_risk_score + rule_weight * rule_risk_score)
        combined_confidence = (ml_weight * ml_confidence + rule_weight * rule_confidence)
        
        # Determine final risk level
        risk_level = self._determine_risk_level(combined_fraud_prob * 100)
        
        # Get recommendation
        recommendation = self._get_recommendation(risk_level, combined_fraud_prob * 100)
        
        # Override recommendation if high-confidence rules triggered
        if high_confidence_rules:
            critical_actions = [r['action'] for r in high_confidence_rules if r['action'] in ['decline', 'block']]
            if critical_actions:
                recommendation = 'decline'
                risk_level = 'critical'
        
        return {
            'fraud_probability': combined_fraud_prob,
            'risk_score': combined_risk_score,
            'confidence': combined_confidence,
            'risk_level': risk_level,
            'recommendation': recommendation,
            'ml_results': ml_results,
            'rule_results': rule_results,
            'triggered_rules_count': len(triggered_rules),
            'high_confidence_rules_count': len(high_confidence_rules),
            'feature_summary': self._get_feature_summary(features) if features else {},
            'explanation': self._generate_explanation(
                ml_results, triggered_rules, combined_fraud_prob, risk_level
            )
        }
    
    def _get_feature_summary(self, features) -> Dict[str, Any]:
        """Generate summary of key features."""
        feature_dict = features.to_dict()
        
        # Get top risk features
        risk_features = {
            k: v for k, v in feature_dict.items()
            if 'risk' in k.lower() or 'anomaly' in k.lower() or 'deviation' in k.lower()
        }
        
        # Sort by value (descending)
        top_risk_features = dict(sorted(risk_features.items(), key=lambda x: x[1], reverse=True)[:10])
        
        return {
            'total_features': len(feature_dict),
            'top_risk_features': top_risk_features,
            'velocity_features_count': len([k for k in feature_dict.keys() if 'velocity' in k]),
            'behavioral_features_count': len([k for k in feature_dict.keys() if 'behavioral' in k]),
            'device_features_count': len([k for k in feature_dict.keys() if 'device' in k]),
        }
    
    def _generate_explanation(
        self,
        ml_results: Dict[str, Any],
        triggered_rules: List[Dict[str, Any]],
        fraud_probability: float,
        risk_level: str
    ) -> Dict[str, Any]:
        """Generate human-readable explanation of the fraud decision."""
        
        explanation = {
            'summary': f"Transaction classified as {risk_level} risk with {fraud_probability:.1%} fraud probability.",
            'key_factors': [],
            'model_contributions': {},
            'rule_contributions': []
        }
        
        # ML model contributions
        model_predictions = ml_results.get('model_predictions', {})
        for model_name, prediction in model_predictions.items():
            if 'error' not in prediction:
                explanation['model_contributions'][model_name] = {
                    'fraud_probability': prediction.get('fraud_probability', 0.0),
                    'confidence': prediction.get('confidence', 0.0)
                }
        
        # Rule contributions
        for rule in triggered_rules:
            explanation['rule_contributions'].append({
                'rule_name': rule['rule_name'],
                'action': rule['action'],
                'confidence': rule['confidence'],
                'conditions_met': len(rule['conditions_met'])
            })
        
        # Key risk factors
        if fraud_probability > 0.7:
            explanation['key_factors'].append("High fraud probability detected by ML models")
        if len(triggered_rules) > 2:
            explanation['key_factors'].append(f"Multiple fraud rules triggered ({len(triggered_rules)})")
        if ml_results.get('ml_confidence', 0) > 0.8:
            explanation['key_factors'].append("High confidence in ML prediction")
        
        return explanation
    
    def _get_basic_ml_prediction(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback ML prediction when ensemble is not available."""
        # Simple rule-based scoring as fallback
        amount = float(transaction_data.get('amount', 0))
        
        # Basic risk scoring
        risk_score = 0.0
        if amount > 1000:
            risk_score += 0.3
        if amount > 5000:
            risk_score += 0.2
        
        # Time-based risk
        try:
            timestamp = datetime.fromisoformat(transaction_data['timestamp'].replace('Z', '+00:00'))
            if timestamp.hour < 6 or timestamp.hour > 22:
                risk_score += 0.1
        except:
            pass
        
        fraud_probability = min(risk_score, 0.9)
        
        return {
            'ml_fraud_probability': fraud_probability,
            'ml_risk_score': fraud_probability,
            'ml_confidence': 0.5,
            'model_predictions': {'fallback': {'fraud_probability': fraud_probability}},
            'ensemble_weights': {'fallback': 1.0},
            'ml_inference_time_ms': 1.0,
            'model_type': 'fallback',
            'models_used': ['fallback']
        }
    
    def _get_fallback_result(self, transaction_data: Dict[str, Any], error: str) -> Dict[str, Any]:
        """Generate fallback result when analysis fails."""
        return {
            'fraud_probability': 0.5,
            'risk_score': 0.5,
            'confidence': 0.0,
            'risk_level': 'unknown',
            'recommendation': 'manual_review',
            'error': error,
            'fallback': True,
            'analysis_timestamp': datetime.now().isoformat(),
            'model_version': '2.0.0-fallback'
        }
    
    def _determine_risk_level(self, fraud_score: float) -> str:
        """Determine risk level based on fraud score."""
        if fraud_score >= 90:
            return 'critical'
        elif fraud_score >= self.high_threshold:
            return 'high'
        elif fraud_score >= self.medium_threshold:
            return 'medium'
        elif fraud_score >= self.low_threshold:
            return 'low'
        else:
            return 'very_low'
    
    def _get_recommendation(self, risk_level: str, fraud_score: float) -> str:
        """Get recommendation based on risk level."""
        recommendations = {
            'very_low': 'approve',
            'low': 'approve',
            'medium': 'manual_review',
            'high': 'decline',
            'critical': 'block'
        }
        return recommendations.get(risk_level, 'manual_review')
    
    def _update_performance_metrics(self, inference_time: float, results: Dict[str, Any]):
        """Update service performance metrics."""
        self.performance_metrics['total_predictions'] += 1
        
        # Update average inference time (exponential moving average)
        alpha = 0.1
        self.performance_metrics['avg_inference_time'] = (
            alpha * inference_time +
            (1 - alpha) * self.performance_metrics['avg_inference_time']
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current service performance metrics."""
        feature_metrics = self.feature_pipeline.get_performance_metrics()
        ensemble_metrics = (
            self.ensemble_engine.get_performance_metrics() 
            if self.ensemble_engine else {}
        )
        
        return {
            'service_metrics': self.performance_metrics,
            'feature_pipeline_metrics': feature_metrics,
            'ensemble_model_metrics': ensemble_metrics,
            'redis_connected': self.redis_client is not None,
            'ensemble_available': self.ensemble_engine is not None
        }
    
    def update_model_performance(
        self,
        actual_fraud: bool,
        predicted_fraud_prob: float,
        model_name: str = 'ensemble'
    ):
        """Update model performance based on actual outcomes."""
        if self.ensemble_engine and hasattr(self.ensemble_engine.ensemble, 'update_model_performance'):
            # Calculate accuracy and confidence metrics
            prediction_correct = (predicted_fraud_prob > 0.5) == actual_fraud
            confidence = abs(predicted_fraud_prob - 0.5) * 2  # Convert to 0-1 scale
            
            self.ensemble_engine.ensemble.update_model_performance(
                model_name=model_name,
                accuracy=1.0 if prediction_correct else 0.0,
                confidence=confidence
            )


# Maintain backward compatibility
class FraudDetectionService(AdvancedFraudDetectionService):
    """Backward compatible fraud detection service."""
    pass