import asyncio
import json
import unittest.mock
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from apps.fraud_detection.models import FraudRule
from apps.fraud_detection.services import AdvancedFraudDetectionService

User = get_user_model()


class TestAdvancedFraudDetectionService(TestCase):
    """Test suite for AdvancedFraudDetectionService."""

    def setUp(self):
        """Set up test fixtures."""
        self.user = User.objects.create_user(
            username="testuser", email="test@example.com", password="testpass123"
        )

        self.sample_transaction = {
            "id": "txn_123",
            "amount": 150.00,
            "currency": "USD",
            "timestamp": "2023-12-01T10:30:00Z",
            "merchant_id": "merchant_456",
            "user_id": str(self.user.id),
            "payment_method": "credit_card",
            "location": {"lat": 40.7128, "lon": -74.0060},
            "device_id": "device_789"
        }

        self.sample_context = {
            "user_transactions": [
                {"amount": 50.0, "timestamp": "2023-11-30T10:00:00Z"},
                {"amount": 75.0, "timestamp": "2023-11-29T15:30:00Z"}
            ],
            "merchant_transactions": [
                {"amount": 200.0, "timestamp": "2023-11-30T12:00:00Z"}
            ],
            "transaction_graph": []
        }

    @override_settings(
        FRAUD_THRESHOLD_LOW=20,
        FRAUD_THRESHOLD_MEDIUM=50,
        FRAUD_THRESHOLD_HIGH=80,
        REDIS_HOST="localhost",
        REDIS_PORT=6379,
        REDIS_DB=0
    )
    def test_service_initialization(self):
        """Test service initialization with settings."""
        service = AdvancedFraudDetectionService()
        
        self.assertEqual(service.low_threshold, 20)
        self.assertEqual(service.medium_threshold, 50)
        self.assertEqual(service.high_threshold, 80)
        self.assertIsNotNone(service.performance_metrics)
        self.assertEqual(service.performance_metrics["total_predictions"], 0)

    @patch('redis.Redis')
    @patch('apps.fraud_detection.services.FeatureEngineeringPipeline')
    @patch('apps.fraud_detection.services.EnsembleInferenceEngine')
    def test_service_initialization_with_dependencies(self, mock_ensemble, mock_pipeline, mock_redis):
        """Test service initialization with mocked dependencies."""
        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance
        
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance
        
        mock_ensemble_instance = Mock()
        mock_ensemble.return_value = mock_ensemble_instance
        
        service = AdvancedFraudDetectionService()
        
        mock_redis.assert_called_once()
        mock_pipeline.assert_called_once_with(
            redis_client=mock_redis_instance,
            enable_caching=True,
            cache_ttl=300,
            max_workers=4
        )
        mock_ensemble.assert_called_once()

    @patch('redis.Redis')
    def test_service_initialization_redis_failure(self, mock_redis):
        """Test service initialization when Redis connection fails."""
        mock_redis.side_effect = Exception("Redis connection failed")
        
        with patch('apps.fraud_detection.services.logger') as mock_logger:
            service = AdvancedFraudDetectionService()
            
            self.assertIsNone(service.redis_client)
            mock_logger.warning.assert_called_once()

    @patch('apps.fraud_detection.services.EnsembleInferenceEngine')
    def test_service_initialization_ensemble_failure(self, mock_ensemble):
        """Test service initialization when ensemble model fails."""
        mock_ensemble.side_effect = Exception("Model loading failed")
        
        with patch('apps.fraud_detection.services.logger') as mock_logger:
            service = AdvancedFraudDetectionService()
            
            self.assertIsNone(service.ensemble_engine)
            mock_logger.warning.assert_called_once()

    @patch('redis.Redis')
    @patch('apps.fraud_detection.services.FeatureEngineeringPipeline')
    @patch('apps.fraud_detection.services.EnsembleInferenceEngine')
    async def test_analyze_transaction_async_success(self, mock_ensemble, mock_pipeline, mock_redis):
        """Test successful asynchronous transaction analysis."""
        # Mock dependencies
        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance
        
        mock_features = Mock()
        mock_features.to_dict.return_value = {"feature_velocity": 0.5, "feature_amount_deviation": 0.3}
        
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.extract_features.return_value = mock_features
        mock_pipeline.return_value = mock_pipeline_instance
        
        mock_ensemble_result = Mock()
        mock_ensemble_result.fraud_probability = 0.7
        mock_ensemble_result.risk_score = 0.6
        mock_ensemble_result.confidence = 0.8
        mock_ensemble_result.model_predictions = {"xgboost": {"fraud_probability": 0.7}}
        mock_ensemble_result.ensemble_weights = {"xgboost": 1.0}
        mock_ensemble_result.inference_time_ms = 50.0
        
        mock_ensemble_instance = Mock()
        mock_ensemble_instance.predict.return_value = mock_ensemble_result
        mock_ensemble.return_value = mock_ensemble_instance
        
        # Create fraud rule
        FraudRule.objects.create(
            name="High Amount Rule",
            description="Flag high amount transactions",
            conditions={
                "amount_threshold": {
                    "type": "threshold",
                    "field": "amount",
                    "operator": ">",
                    "threshold": 100.0
                }
            },
            action="flag",
            owner=self.user
        )
        
        service = AdvancedFraudDetectionService()
        result = await service.analyze_transaction_async(
            self.sample_transaction, self.user, self.sample_context
        )
        
        # Verify result structure
        self.assertIn("fraud_probability", result)
        self.assertIn("risk_score", result)
        self.assertIn("confidence", result)
        self.assertIn("risk_level", result)
        self.assertIn("recommendation", result)
        self.assertIn("ml_results", result)
        self.assertIn("rule_results", result)
        self.assertIn("performance_metrics", result)
        self.assertIn("analysis_timestamp", result)
        
        # Verify performance metrics
        self.assertIn("total_time_ms", result["performance_metrics"])
        self.assertIn("feature_extraction_time_ms", result["performance_metrics"])
        self.assertIn("ml_prediction_time_ms", result["performance_metrics"])
        self.assertIn("rules_evaluation_time_ms", result["performance_metrics"])

    @patch('redis.Redis')
    def test_analyze_transaction_sync_wrapper(self, mock_redis):
        """Test synchronous wrapper for transaction analysis."""
        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance
        
        service = AdvancedFraudDetectionService()
        
        # Mock the async method
        async def mock_async_analyze(transaction_data, user, context):
            return {"fraud_probability": 0.5, "risk_level": "medium"}
        
        with patch.object(service, 'analyze_transaction_async', side_effect=mock_async_analyze):
            result = service.analyze_transaction(
                self.sample_transaction, self.user, self.sample_context
            )
            
            self.assertEqual(result["fraud_probability"], 0.5)
            self.assertEqual(result["risk_level"], "medium")

    @patch('redis.Redis')
    def test_analyze_transaction_sync_wrapper_exception(self, mock_redis):
        """Test synchronous wrapper exception handling."""
        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance
        
        service = AdvancedFraudDetectionService()
        
        # Mock the async method to raise exception
        async def mock_async_analyze_error(transaction_data, user, context):
            raise Exception("Analysis failed")
        
        with patch.object(service, 'analyze_transaction_async', side_effect=mock_async_analyze_error):
            with patch('apps.fraud_detection.services.logger') as mock_logger:
                result = service.analyze_transaction(
                    self.sample_transaction, self.user, self.sample_context
                )
                
                self.assertTrue(result.get("fallback", False))
                self.assertIn("error", result)
                mock_logger.error.assert_called_once()

    @patch('redis.Redis')
    @patch('apps.fraud_detection.services.FeatureEngineeringPipeline')
    async def test_extract_features_async_success(self, mock_pipeline, mock_redis):
        """Test successful feature extraction."""
        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance
        
        mock_features = Mock()
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.extract_features.return_value = mock_features
        mock_pipeline.return_value = mock_pipeline_instance
        
        service = AdvancedFraudDetectionService()
        result = await service._extract_features_async(
            self.sample_transaction, self.sample_context
        )
        
        self.assertEqual(result, mock_features)
        mock_pipeline_instance.extract_features.assert_called_once_with(
            transaction=self.sample_transaction,
            user_transactions=self.sample_context["user_transactions"],
            merchant_transactions=self.sample_context["merchant_transactions"],
            use_cache=True
        )

    @patch('redis.Redis')
    @patch('apps.fraud_detection.services.FeatureEngineeringPipeline')
    async def test_extract_features_async_exception(self, mock_pipeline, mock_redis):
        """Test feature extraction exception handling."""
        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance
        
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.extract_features.side_effect = Exception("Feature extraction failed")
        mock_pipeline.return_value = mock_pipeline_instance
        
        service = AdvancedFraudDetectionService()
        
        with patch('apps.fraud_detection.services.logger') as mock_logger:
            result = await service._extract_features_async(
                self.sample_transaction, self.sample_context
            )
            
            self.assertIsNone(result)
            mock_logger.error.assert_called_once()

    @patch('redis.Redis')
    @patch('apps.fraud_detection.services.EnsembleInferenceEngine')
    async def test_run_ml_prediction_async_with_ensemble(self, mock_ensemble, mock_redis):
        """Test ML prediction with ensemble model."""
        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance
        
        mock_ensemble_result = Mock()
        mock_ensemble_result.fraud_probability = 0.8
        mock_ensemble_result.risk_score = 0.7
        mock_ensemble_result.confidence = 0.9
        mock_ensemble_result.model_predictions = {
            "xgboost": {"fraud_probability": 0.8, "confidence": 0.9},
            "lstm": {"fraud_probability": 0.7, "confidence": 0.8}
        }
        mock_ensemble_result.ensemble_weights = {"xgboost": 0.6, "lstm": 0.4}
        mock_ensemble_result.inference_time_ms = 75.0
        
        mock_ensemble_instance = Mock()
        mock_ensemble_instance.predict.return_value = mock_ensemble_result
        mock_ensemble.return_value = mock_ensemble_instance
        
        mock_features = Mock()
        
        service = AdvancedFraudDetectionService()
        result = await service._run_ml_prediction_async(
            self.sample_transaction, mock_features, self.sample_context
        )
        
        self.assertEqual(result["ml_fraud_probability"], 0.8)
        self.assertEqual(result["ml_risk_score"], 0.7)
        self.assertEqual(result["ml_confidence"], 0.9)
        self.assertEqual(result["model_type"], "ensemble")
        self.assertIn("xgboost", result["models_used"])
        self.assertIn("lstm", result["models_used"])

    @patch('redis.Redis')
    async def test_run_ml_prediction_async_fallback(self, mock_redis):
        """Test ML prediction fallback when ensemble is not available."""
        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance
        
        service = AdvancedFraudDetectionService()
        service.ensemble_engine = None
        
        result = await service._run_ml_prediction_async(
            self.sample_transaction, None, self.sample_context
        )
        
        self.assertEqual(result["model_type"], "fallback")
        self.assertIn("ml_fraud_probability", result)
        self.assertIn("fallback", result["models_used"])

    @patch('redis.Redis')
    async def test_apply_rules_async_success(self, mock_redis):
        """Test successful rule application."""
        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance
        
        # Create test rules
        rule1 = FraudRule.objects.create(
            name="Amount Rule",
            conditions={
                "amount_check": {
                    "type": "threshold",
                    "field": "amount",
                    "operator": ">",
                    "threshold": 100.0
                }
            },
            action="flag",
            owner=self.user
        )
        
        rule2 = FraudRule.objects.create(
            name="Time Rule",
            conditions={
                "time_check": {
                    "type": "categorical",
                    "field": "payment_method",
                    "values": ["credit_card", "debit_card"]
                }
            },
            action="alert",
            owner=self.user
        )
        
        mock_features = Mock()
        
        service = AdvancedFraudDetectionService()
        result = await service._apply_rules_async(
            self.sample_transaction, self.user, mock_features
        )
        
        self.assertEqual(len(result), 2)
        self.assertIn("rule_id", result[0])
        self.assertIn("rule_name", result[0])
        self.assertIn("triggered", result[0])
        self.assertIn("confidence", result[0])

    @patch('redis.Redis')
    async def test_apply_rules_async_exception(self, mock_redis):
        """Test rule application exception handling."""
        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance
        
        service = AdvancedFraudDetectionService()
        
        # Mock FraudRule.objects.filter to raise exception
        with patch('apps.fraud_detection.services.FraudRule.objects.filter') as mock_filter:
            mock_filter.side_effect = Exception("Database error")
            
            with patch('apps.fraud_detection.services.logger') as mock_logger:
                result = await service._apply_rules_async(
                    self.sample_transaction, self.user, None
                )
                
                self.assertEqual(result, [])
                mock_logger.error.assert_called_once()

    @patch('redis.Redis')
    async def test_evaluate_rule_async_threshold_condition(self, mock_redis):
        """Test rule evaluation with threshold conditions."""
        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance
        
        rule = FraudRule.objects.create(
            name="Amount Rule",
            conditions={
                "amount_check": {
                    "type": "threshold",
                    "field": "amount",
                    "operator": ">",
                    "threshold": 100.0
                },
                "_meta": {"required_conditions": 1}
            },
            action="flag",
            owner=self.user
        )
        
        mock_features = Mock()
        mock_features.to_dict.return_value = {}
        
        service = AdvancedFraudDetectionService()
        result = await service._evaluate_rule_async(
            rule, self.sample_transaction, mock_features
        )
        
        self.assertTrue(result["triggered"])  # 150 > 100
        self.assertGreater(result["confidence"], 0)
        self.assertEqual(len(result["conditions_met"]), 1)

    @patch('redis.Redis')
    async def test_evaluate_rule_async_categorical_condition(self, mock_redis):
        """Test rule evaluation with categorical conditions."""
        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance
        
        rule = FraudRule.objects.create(
            name="Payment Method Rule",
            conditions={
                "payment_check": {
                    "type": "categorical",
                    "field": "payment_method",
                    "values": ["credit_card", "paypal"]
                },
                "_meta": {"required_conditions": 1}
            },
            action="alert",
            owner=self.user
        )
        
        mock_features = Mock()
        mock_features.to_dict.return_value = {}
        
        service = AdvancedFraudDetectionService()
        result = await service._evaluate_rule_async(
            rule, self.sample_transaction, mock_features
        )
        
        self.assertTrue(result["triggered"])  # credit_card is in allowed values
        self.assertEqual(result["confidence"], 1.0)

    def test_evaluate_condition_threshold_operators(self):
        """Test threshold condition evaluation with different operators."""
        service = AdvancedFraudDetectionService()
        
        # Test greater than
        result = service._evaluate_condition(
            {"type": "threshold", "field": "amount", "operator": ">", "threshold": 100.0},
            {"amount": 150.0},
            {}
        )
        self.assertTrue(result["met"])
        
        # Test less than
        result = service._evaluate_condition(
            {"type": "threshold", "field": "amount", "operator": "<", "threshold": 100.0},
            {"amount": 50.0},
            {}
        )
        self.assertTrue(result["met"])
        
        # Test equals
        result = service._evaluate_condition(
            {"type": "threshold", "field": "amount", "operator": "==", "threshold": 150.0},
            {"amount": 150.0},
            {}
        )
        self.assertTrue(result["met"])

    def test_evaluate_condition_feature_field(self):
        """Test condition evaluation with feature fields."""
        service = AdvancedFraudDetectionService()
        
        result = service._evaluate_condition(
            {"type": "threshold", "field": "feature_velocity", "operator": ">", "threshold": 0.5},
            {"amount": 100.0},
            {"feature_velocity": 0.8}
        )
        
        self.assertTrue(result["met"])
        self.assertEqual(result["value"], 0.8)

    def test_evaluate_condition_categorical(self):
        """Test categorical condition evaluation."""
        service = AdvancedFraudDetectionService()
        
        # Test matching value
        result = service._evaluate_condition(
            {"type": "categorical", "field": "status", "values": ["active", "pending"]},
            {"status": "active"},
            {}
        )
        self.assertTrue(result["met"])
        self.assertEqual(result["confidence"], 1.0)
        
        # Test non-matching value
        result = service._evaluate_condition(
            {"type": "categorical", "field": "status", "values": ["active", "pending"]},
            {"status": "inactive"},
            {}
        )
        self.assertFalse(result["met"])
        self.assertEqual(result["confidence"], 0.0)

    def test_combine_analysis_results(self):
        """Test combining ML and rule analysis results."""
        service = AdvancedFraudDetectionService()
        
        ml_results = {
            "ml_fraud_probability": 0.7,
            "ml_risk_score": 0.6,
            "ml_confidence": 0.8,
            "model_predictions": {"xgboost": {"fraud_probability": 0.7}}
        }
        
        rule_results = [
            {
                "rule_name": "High Amount",
                "triggered": True,
                "confidence": 0.9,
                "action": "flag"
            },
            {
                "rule_name": "Suspicious Location",
                "triggered": True,
                "confidence": 0.7,
                "action": "decline"
            }
        ]
        
        mock_features = Mock()
        mock_features.to_dict.return_value = {"feature_amount": 150.0}
        
        result = service._combine_analysis_results(
            ml_results, rule_results, mock_features, self.sample_transaction
        )
        
        self.assertIn("fraud_probability", result)
        self.assertIn("risk_score", result)
        self.assertIn("confidence", result)
        self.assertIn("risk_level", result)
        self.assertIn("recommendation", result)
        self.assertEqual(result["triggered_rules_count"], 2)
        self.assertIn("explanation", result)

    def test_combine_analysis_results_high_confidence_override(self):
        """Test recommendation override for high-confidence rules."""
        service = AdvancedFraudDetectionService()
        
        ml_results = {
            "ml_fraud_probability": 0.3,
            "ml_risk_score": 0.3,
            "ml_confidence": 0.5
        }
        
        rule_results = [
            {
                "rule_name": "Critical Rule",
                "triggered": True,
                "confidence": 0.95,
                "action": "decline"
            }
        ]
        
        result = service._combine_analysis_results(
            ml_results, rule_results, None, self.sample_transaction
        )
        
        self.assertEqual(result["recommendation"], "decline")
        self.assertEqual(result["risk_level"], "critical")

    def test_get_feature_summary(self):
        """Test feature summary generation."""
        service = AdvancedFraudDetectionService()
        
        mock_features = Mock()
        mock_features.to_dict.return_value = {
            "feature_velocity_risk": 0.8,
            "feature_amount_deviation": 0.6,
            "feature_behavioral_score": 0.4,
            "device_fingerprint": 0.2,
            "velocity_last_hour": 0.9,
            "normal_feature": 0.1
        }
        
        result = service._get_feature_summary(mock_features)
        
        self.assertIn("total_features", result)
        self.assertIn("top_risk_features", result)
        self.assertIn("velocity_features_count", result)
        self.assertIn("behavioral_features_count", result)
        self.assertIn("device_features_count", result)
        
        self.assertEqual(result["total_features"], 6)
        self.assertEqual(result["velocity_features_count"], 2)
        self.assertEqual(result["behavioral_features_count"], 1)
        self.assertEqual(result["device_features_count"], 1)

    def test_generate_explanation(self):
        """Test fraud decision explanation generation."""
        service = AdvancedFraudDetectionService()
        
        ml_results = {
            "model_predictions": {
                "xgboost": {"fraud_probability": 0.8, "confidence": 0.9},
                "lstm": {"fraud_probability": 0.7, "confidence": 0.8}
            }
        }
        
        triggered_rules = [
            {
                "rule_name": "High Amount Rule",
                "action": "flag",
                "confidence": 0.9,
                "conditions_met": [{"condition": "amount", "value": 150}]
            }
        ]
        
        result = service._generate_explanation(
            ml_results, triggered_rules, 0.75, "high"
        )
        
        self.assertIn("summary", result)
        self.assertIn("key_factors", result)
        self.assertIn("model_contributions", result)
        self.assertIn("rule_contributions", result)
        
        self.assertIn("high risk", result["summary"])
        self.assertEqual(len(result["model_contributions"]), 2)
        self.assertEqual(len(result["rule_contributions"]), 1)

    def test_get_basic_ml_prediction(self):
        """Test basic ML prediction fallback."""
        service = AdvancedFraudDetectionService()
        
        # Test high amount transaction
        high_amount_transaction = {
            "amount": 2000.0,
            "timestamp": "2023-12-01T02:00:00Z"
        }
        
        result = service._get_basic_ml_prediction(high_amount_transaction)
        
        self.assertEqual(result["model_type"], "fallback")
        self.assertGreater(result["ml_fraud_probability"], 0.5)  # High amount + night time
        self.assertIn("fallback", result["models_used"])

    def test_get_basic_ml_prediction_timestamp_parsing_error(self):
        """Test basic ML prediction with invalid timestamp."""
        service = AdvancedFraudDetectionService()
        
        transaction_with_bad_timestamp = {
            "amount": 1000.0,
            "timestamp": "invalid-timestamp"
        }
        
        with patch('apps.fraud_detection.services.logger') as mock_logger:
            result = service._get_basic_ml_prediction(transaction_with_bad_timestamp)
            
            self.assertEqual(result["model_type"], "fallback")
            mock_logger.warning.assert_called_once()

    def test_get_fallback_result(self):
        """Test fallback result generation."""
        service = AdvancedFraudDetectionService()
        
        result = service._get_fallback_result(
            self.sample_transaction, "Test error message"
        )
        
        self.assertEqual(result["fraud_probability"], 0.5)
        self.assertEqual(result["risk_level"], "unknown")
        self.assertEqual(result["recommendation"], "manual_review")
        self.assertEqual(result["error"], "Test error message")
        self.assertTrue(result["fallback"])
        self.assertIn("analysis_timestamp", result)

    def test_determine_risk_level(self):
        """Test risk level determination."""
        service = AdvancedFraudDetectionService()
        service.low_threshold = 20
        service.medium_threshold = 50
        service.high_threshold = 80
        
        self.assertEqual(service._determine_risk_level(95), "critical")
        self.assertEqual(service._determine_risk_level(85), "high")
        self.assertEqual(service._determine_risk_level(60), "medium")
        self.assertEqual(service._determine_risk_level(30), "low")
        self.assertEqual(service._determine_risk_level(10), "very_low")

    def test_get_recommendation(self):
        """Test recommendation based on risk level."""
        service = AdvancedFraudDetectionService()
        
        self.assertEqual(service._get_recommendation("very_low", 10), "approve")
        self.assertEqual(service._get_recommendation("low", 30), "approve")
        self.assertEqual(service._get_recommendation("medium", 60), "manual_review")
        self.assertEqual(service._get_recommendation("high", 85), "decline")
        self.assertEqual(service._get_recommendation("critical", 95), "block")
        self.assertEqual(service._get_recommendation("unknown", 50), "manual_review")

    @patch('redis.Redis')
    def test_update_performance_metrics(self, mock_redis):
        """Test performance metrics update."""
        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance
        
        service = AdvancedFraudDetectionService()
        initial_predictions = service.performance_metrics["total_predictions"]
        initial_avg_time = service.performance_metrics["avg_inference_time"]
        
        service._update_performance_metrics(100.0, {"fraud_probability": 0.7})
        
        self.assertEqual(
            service.performance_metrics["total_predictions"], 
            initial_predictions + 1
        )
        self.assertGreater(
            service.performance_metrics["avg_inference_time"], 
            initial_avg_time
        )

    @patch('redis.Redis')
    @patch('apps.fraud_detection.services.FeatureEngineeringPipeline')
    @patch('apps.fraud_detection.services.EnsembleInferenceEngine')
    def test_get_performance_metrics(self, mock_ensemble, mock_pipeline, mock_redis):
        """Test performance metrics retrieval."""
        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance
        
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.get_performance_metrics.return_value = {"feature_time": 50.0}
        mock_pipeline.return_value = mock_pipeline_instance
        
        mock_ensemble_instance = Mock()
        mock_ensemble_instance.get_performance_metrics.return_value = {"model_accuracy": 0.85}
        mock_ensemble.return_value = mock_ensemble_instance
        
        service = AdvancedFraudDetectionService()
        metrics = service.get_performance_metrics()
        
        self.assertIn("service_metrics", metrics)
        self.assertIn("feature_pipeline_metrics", metrics)
        self.assertIn("ensemble_model_metrics", metrics)
        self.assertIn("redis_connected", metrics)
        self.assertIn("ensemble_available", metrics)
        
        self.assertTrue(metrics["redis_connected"])
        self.assertTrue(metrics["ensemble_available"])

    @patch('redis.Redis')
    @patch('apps.fraud_detection.services.EnsembleInferenceEngine')
    def test_update_model_performance(self, mock_ensemble, mock_redis):
        """Test model performance update."""
        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance
        
        mock_ensemble_instance = Mock()
        mock_ensemble_instance.ensemble = Mock()
        mock_ensemble_instance.ensemble.update_model_performance = Mock()
        mock_ensemble.return_value = mock_ensemble_instance
        
        service = AdvancedFraudDetectionService()
        service.update_model_performance(
            actual_fraud=True,
            predicted_fraud_prob=0.8,
            model_name="xgboost"
        )
        
        mock_ensemble_instance.ensemble.update_model_performance.assert_called_once()
        call_args = mock_ensemble_instance.ensemble.update_model_performance.call_args
        self.assertEqual(call_args[1]["model_name"], "xgboost")
        self.assertEqual(call_args[1]["accuracy"], 1.0)  # Prediction was correct
        self.assertGreater(call_args[1]["confidence"], 0)

    @patch('redis.Redis')
    def test_backward_compatibility_service(self, mock_redis):
        """Test backward compatible FraudDetectionService."""
        from apps.fraud_detection.services import FraudDetectionService
        
        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance
        
        service = FraudDetectionService()
        self.assertIsInstance(service, AdvancedFraudDetectionService)

    @patch('redis.Redis')
    @patch('apps.fraud_detection.services.FeatureEngineeringPipeline')
    @patch('apps.fraud_detection.services.EnsembleInferenceEngine')
    async def test_analyze_transaction_async_exception_handling(self, mock_ensemble, mock_pipeline, mock_redis):
        """Test exception handling in async transaction analysis."""
        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance
        
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.extract_features.side_effect = Exception("Feature extraction failed")
        mock_pipeline.return_value = mock_pipeline_instance
        
        mock_ensemble_instance = Mock()
        mock_ensemble.return_value = mock_ensemble_instance
        
        service = AdvancedFraudDetectionService()
        
        with patch('apps.fraud_detection.services.logger') as mock_logger:
            result = await service.analyze_transaction_async(
                self.sample_transaction, self.user, self.sample_context
            )
            
            self.assertTrue(result.get("fallback", False))
            self.assertIn("error", result)
            mock_logger.error.assert_called_once()

    def test_numpy_usage_in_conditions_met_confidence(self):
        """Test numpy usage in confidence calculation."""
        service = AdvancedFraudDetectionService()
        
        # Test with empty conditions_met list
        mock_features = Mock()
        mock_features.to_dict.return_value = {}
        
        rule = Mock()
        rule.conditions = {"test": {"type": "threshold", "field": "amount", "operator": "<", "threshold": 100}}
        
        with patch('apps.fraud_detection.services.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 12, 1, 10, 0, 0)
            
            # This should not raise any numpy-related errors
            result = asyncio.run(service._evaluate_rule_async(rule, self.sample_transaction, mock_features))
            
            # When conditions_met is empty, confidence should be 0.0
            self.assertEqual(result["confidence"], 0.0)


@pytest.mark.integration
class TestAdvancedFraudDetectionServiceIntegration(TestCase):
    """Integration tests for AdvancedFraudDetectionService."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.user = User.objects.create_user(
            username="integrationuser",
            email="integration@example.com",
            password="testpass123"
        )

    @override_settings(
        FRAUD_THRESHOLD_LOW=20,
        FRAUD_THRESHOLD_MEDIUM=50,
        FRAUD_THRESHOLD_HIGH=80
    )
    @patch('redis.Redis')
    def test_end_to_end_fraud_analysis(self, mock_redis):
        """Test end-to-end fraud analysis flow."""
        # Mock Redis to prevent actual connection
        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance
        
        # Create multiple fraud rules
        FraudRule.objects.create(
            name="High Amount Rule",
            description="Flag transactions over $500",
            conditions={
                "amount_threshold": {
                    "type": "threshold",
                    "field": "amount",
                    "operator": ">",
                    "threshold": 500.0
                }
            },
            action="flag",
            owner=self.user
        )
        
        FraudRule.objects.create(
            name="Night Time Rule",
            description="Flag late night transactions",
            conditions={
                "payment_method_check": {
                    "type": "categorical",
                    "field": "payment_method",
                    "values": ["credit_card"]
                }
            },
            action="alert",
            owner=self.user
        )
        
        # High-risk transaction
        high_risk_transaction = {
            "id": "txn_high_risk",
            "amount": 1500.00,  # Triggers amount rule
            "currency": "USD",
            "timestamp": "2023-12-01T23:30:00Z",  # Night time
            "merchant_id": "merchant_suspicious",
            "user_id": str(self.user.id),
            "payment_method": "credit_card",  # Triggers payment method rule
            "location": {"lat": 40.7128, "lon": -74.0060}
        }
        
        service = AdvancedFraudDetectionService()
        
        # Mock feature pipeline and ensemble to avoid external dependencies
        with patch.object(service.feature_pipeline, 'extract_features') as mock_extract:
            mock_features = Mock()
            mock_features.to_dict.return_value = {
                "feature_amount_risk": 0.9,
                "feature_velocity": 0.7,
                "feature_behavioral_deviation": 0.8
            }
            mock_extract.return_value = mock_features
            
            result = service.analyze_transaction(high_risk_transaction, self.user)
            
            # Verify comprehensive result structure
            required_fields = [
                "fraud_probability", "risk_score", "confidence", "risk_level",
                "recommendation", "ml_results", "rule_results", "performance_metrics",
                "analysis_timestamp", "model_version", "service_version"
            ]
            
            for field in required_fields:
                self.assertIn(field, result, f"Missing required field: {field}")
            
            # Verify that rules were triggered
            self.assertGreater(result["triggered_rules_count"], 0)
            
            # Verify performance metrics are present
            self.assertIn("total_time_ms", result["performance_metrics"])
            self.assertGreater(result["performance_metrics"]["total_time_ms"], 0)

    @override_settings(
        FRAUD_THRESHOLD_LOW=20,
        FRAUD_THRESHOLD_MEDIUM=50,
        FRAUD_THRESHOLD_HIGH=80
    )
    @patch('redis.Redis')
    def test_low_risk_transaction_analysis(self, mock_redis):
        """Test analysis of low-risk transaction."""
        # Mock Redis
        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance
        
        # Low-risk transaction
        low_risk_transaction = {
            "id": "txn_low_risk",
            "amount": 25.00,  # Small amount
            "currency": "USD",
            "timestamp": "2023-12-01T14:30:00Z",  # Daytime
            "merchant_id": "merchant_trusted",
            "user_id": str(self.user.id),
            "payment_method": "debit_card",
            "location": {"lat": 40.7128, "lon": -74.0060}
        }
        
        service = AdvancedFraudDetectionService()
        
        # Mock dependencies to simulate low-risk scenario
        with patch.object(service.feature_pipeline, 'extract_features') as mock_extract:
            mock_features = Mock()
            mock_features.to_dict.return_value = {
                "feature_amount_risk": 0.1,
                "feature_velocity": 0.2,
                "feature_behavioral_deviation": 0.1
            }
            mock_extract.return_value = mock_features
            
            result = service.analyze_transaction(low_risk_transaction, self.user)
            
            # Should be low risk
            self.assertIn(result["risk_level"], ["very_low", "low"])
            self.assertIn(result["recommendation"], ["approve"])
            self.assertLess(result["fraud_probability"], 0.5)

    @patch('redis.Redis')
    def test_service_resilience_with_failures(self, mock_redis):
        """Test service resilience when components fail."""
        # Mock Redis to fail
        mock_redis.side_effect = Exception("Redis unavailable")
        
        transaction = {
            "id": "txn_resilience_test",
            "amount": 100.00,
            "currency": "USD",
            "timestamp": "2023-12-01T12:00:00Z",
            "merchant_id": "merchant_test",
            "user_id": str(self.user.id),
            "payment_method": "credit_card"
        }
        
        # Service should still initialize and work with fallbacks
        service = AdvancedFraudDetectionService()
        result = service.analyze_transaction(transaction, self.user)
        
        # Should get a valid result even with component failures
        self.assertIsInstance(result, dict)
        self.assertIn("fraud_probability", result)
        self.assertIn("risk_level", result)
        self.assertIn("recommendation", result)