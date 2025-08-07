from datetime import timedelta
from decimal import Decimal
from django.test import TestCase
from django.contrib.auth import get_user_model
from django.utils import timezone
from apps.transactions.models import Transaction
from .models import Alert, AlertRule
from .services import AlertGenerator

User = get_user_model()


class AlertGeneratorTests(TestCase):
    """Test cases for AlertGenerator service."""
    
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        
        self.transaction = Transaction.objects.create(
            transaction_id='txn_123',
            user_id='user_456',
            amount=Decimal('500.00'),
            currency='USD',
            merchant_id='merchant_789',
            payment_method='credit_card',
            ip_address='192.168.1.1',
            timestamp=timezone.now(),
            owner=self.user
        )
        
        self.fraud_result = {
            'fraud_probability': 0.75,
            'risk_score': 0.8,
            'risk_level': 'high',
            'confidence': 0.9,
            'ml_results': {
                'ml_confidence': 0.85,
                'model_predictions': {
                    'xgboost': {'fraud_probability': 0.7},
                    'neural_net': {'fraud_probability': 0.8}
                }
            },
            'rule_results': []
        }
        
        self.alert_generator = AlertGenerator()
    
    def test_create_alert_context(self):
        """Test alert context creation."""
        context = self.alert_generator._create_alert_context(
            self.transaction, self.fraud_result
        )
        
        self.assertEqual(context['transaction_id'], 'txn_123')
        self.assertEqual(context['amount'], 500.0)
        self.assertEqual(context['fraud_score'], 0.75)
        self.assertEqual(context['risk_level'], 'high')
        self.assertIn('hour_of_day', context)
        self.assertIn('is_weekend', context)
        self.assertIn('amount_category', context)
    
    def test_calculate_derived_features(self):
        """Test derived feature calculation."""
        features = self.alert_generator._calculate_derived_features(
            self.transaction, self.fraud_result
        )
        
        self.assertIn('hour_of_day', features)
        self.assertIn('day_of_week', features)
        self.assertIn('is_high_amount', features)
        self.assertIn('amount_category', features)
        self.assertIn('fraud_score_category', features)
        self.assertIn('ml_confidence', features)
        self.assertIn('model_agreement', features)
    
    def test_categorize_amount(self):
        """Test amount categorization."""
        self.assertEqual(self.alert_generator._categorize_amount(5.0), 'micro')
        self.assertEqual(self.alert_generator._categorize_amount(50.0), 'small')
        self.assertEqual(self.alert_generator._categorize_amount(500.0), 'medium')
        self.assertEqual(self.alert_generator._categorize_amount(5000.0), 'large')
        self.assertEqual(self.alert_generator._categorize_amount(50000.0), 'very_large')
    
    def test_categorize_fraud_score(self):
        """Test fraud score categorization."""
        self.assertEqual(self.alert_generator._categorize_fraud_score(0.1), 'very_low')
        self.assertEqual(self.alert_generator._categorize_fraud_score(0.3), 'low')
        self.assertEqual(self.alert_generator._categorize_fraud_score(0.5), 'medium')
        self.assertEqual(self.alert_generator._categorize_fraud_score(0.7), 'high')
        self.assertEqual(self.alert_generator._categorize_fraud_score(0.9), 'very_high')
    
    def test_calculate_model_agreement(self):
        """Test model agreement calculation."""
        ml_results = {
            'model_predictions': {
                'model1': {'fraud_probability': 0.7},
                'model2': {'fraud_probability': 0.8},
                'model3': {'fraud_probability': 0.75}
            }
        }
        
        agreement = self.alert_generator._calculate_model_agreement(ml_results)
        self.assertGreater(agreement, 0.0)
        self.assertLessEqual(agreement, 1.0)
    
    def test_evaluate_threshold_condition(self):
        """Test threshold condition evaluation."""
        # Greater than condition
        condition = {'threshold': 0.5, 'operator': '>'}
        self.assertTrue(self.alert_generator._evaluate_threshold_condition(condition, 0.7))
        self.assertFalse(self.alert_generator._evaluate_threshold_condition(condition, 0.3))
        
        # Less than condition
        condition = {'threshold': 1000, 'operator': '<'}
        self.assertTrue(self.alert_generator._evaluate_threshold_condition(condition, 500))
        self.assertFalse(self.alert_generator._evaluate_threshold_condition(condition, 1500))
        
        # Equal condition
        condition = {'threshold': 100, 'operator': '=='}
        self.assertTrue(self.alert_generator._evaluate_threshold_condition(condition, 100))
        self.assertFalse(self.alert_generator._evaluate_threshold_condition(condition, 200))
    
    def test_evaluate_categorical_condition(self):
        """Test categorical condition evaluation."""
        condition = {'values': ['credit_card', 'debit_card']}
        
        self.assertTrue(self.alert_generator._evaluate_categorical_condition(condition, 'credit_card'))
        self.assertFalse(self.alert_generator._evaluate_categorical_condition(condition, 'paypal'))
    
    def test_evaluate_range_condition(self):
        """Test range condition evaluation."""
        condition = {'min': 100, 'max': 1000}
        
        self.assertTrue(self.alert_generator._evaluate_range_condition(condition, 500))
        self.assertFalse(self.alert_generator._evaluate_range_condition(condition, 50))
        self.assertFalse(self.alert_generator._evaluate_range_condition(condition, 1500))
    
    def test_get_applicable_rules(self):
        """Test getting applicable rules."""
        # Create test rules
        rule1 = AlertRule.objects.create(
            name='High Fraud Score Rule',
            fraud_score_threshold=0.7,
            alert_type='high_fraud',
            severity='high',
            action='alert',
            owner=self.user
        )
        
        rule2 = AlertRule.objects.create(
            name='High Amount Rule',
            amount_threshold=Decimal('1000.00'),
            alert_type='high_amount',
            severity='medium',
            action='alert',
            owner=self.user
        )
        
        alert_context = {
            'fraud_score': 0.75,
            'amount': 500.0
        }
        
        applicable_rules = self.alert_generator._get_applicable_rules(self.user, alert_context)
        
        # Should include rule1 (fraud score threshold met) but not rule2 (amount threshold not met)
        rule_names = [rule.name for rule in applicable_rules]
        self.assertIn('High Fraud Score Rule', rule_names)
        self.assertNotIn('High Amount Rule', rule_names)
    
    def test_evaluate_rule_simple(self):
        """Test simple rule evaluation."""
        rule = AlertRule.objects.create(
            name='Simple Rule',
            conditions={
                'fraud_score_check': {
                    'type': 'threshold',
                    'field': 'fraud_score',
                    'operator': '>',
                    'threshold': 0.5
                }
            },
            alert_type='simple_alert',
            severity='medium',
            action='alert',
            owner=self.user
        )
        
        alert_context = {'fraud_score': 0.75}
        
        result = self.alert_generator._evaluate_rule(rule, alert_context)
        self.assertTrue(result)
        
        # Test with lower fraud score
        alert_context = {'fraud_score': 0.3}
        result = self.alert_generator._evaluate_rule(rule, alert_context)
        self.assertFalse(result)
    
    def test_evaluate_rule_multiple_conditions(self):
        """Test rule evaluation with multiple conditions."""
        rule = AlertRule.objects.create(
            name='Complex Rule',
            conditions={
                'fraud_score_check': {
                    'type': 'threshold',
                    'field': 'fraud_score',
                    'operator': '>',
                    'threshold': 0.5
                },
                'amount_check': {
                    'type': 'threshold',
                    'field': 'amount',
                    'operator': '>',
                    'threshold': 100
                },
                '_meta': {
                    'required_conditions': 2
                }
            },
            alert_type='complex_alert',
            severity='high',
            action='alert',
            owner=self.user
        )
        
        # Both conditions met
        alert_context = {'fraud_score': 0.75, 'amount': 500.0}
        result = self.alert_generator._evaluate_rule(rule, alert_context)
        self.assertTrue(result)
        
        # Only one condition met
        alert_context = {'fraud_score': 0.75, 'amount': 50.0}
        result = self.alert_generator._evaluate_rule(rule, alert_context)
        self.assertFalse(result)
    
    def test_create_alert(self):
        """Test alert creation."""
        rule = AlertRule.objects.create(
            name='Test Rule',
            alert_type='test_alert',
            severity='medium',
            action='alert',
            owner=self.user
        )
        
        alert_context = {
            'transaction_id': 'txn_123',
            'fraud_score': 0.75,
            'risk_level': 'high',
            'amount': 500.0,
            'currency': 'USD',
            'context_data': {'test': 'data'}
        }
        
        alert = self.alert_generator._create_alert(
            rule, self.transaction, alert_context, self.user
        )
        
        self.assertIsNotNone(alert)
        self.assertEqual(alert.alert_type, 'test_alert')
        self.assertEqual(alert.severity, 'medium')
        self.assertEqual(alert.fraud_score, 0.75)
        self.assertEqual(alert.owner, self.user)
        self.assertEqual(alert.rule_triggered, rule)
    
    def test_generate_alert_title(self):
        """Test alert title generation."""
        rule = AlertRule.objects.create(
            name='Test Rule',
            alert_type='high_risk_transaction',
            severity='high',
            action='alert',
            owner=self.user
        )
        
        # High fraud score
        alert_context = {'fraud_score': 0.85, 'amount': 500.0}
        title = self.alert_generator._generate_alert_title(rule, alert_context)
        self.assertIn('HIGH RISK', title)
        
        # Large amount
        alert_context = {'fraud_score': 0.5, 'amount': 15000.0}
        title = self.alert_generator._generate_alert_title(rule, alert_context)
        self.assertIn('LARGE AMOUNT', title)
    
    def test_extract_risk_factors(self):
        """Test risk factor extraction."""
        alert_context = {
            'fraud_score': 0.85,
            'amount': 1500.0,
            'is_night_time': True,
            'is_weekend': True,
            'ml_results': {'ml_confidence': 0.9},
            'rule_results': [
                {'triggered': True},
                {'triggered': True},
                {'triggered': False}
            ]
        }
        
        risk_factors = self.alert_generator._extract_risk_factors(alert_context)
        
        self.assertIn('high_fraud_score', risk_factors)
        self.assertIn('high_amount', risk_factors)
        self.assertIn('night_transaction', risk_factors)
        self.assertIn('weekend_transaction', risk_factors)
        self.assertIn('high_ml_confidence', risk_factors)
        self.assertIn('multiple_rules_triggered', risk_factors)
    
    def test_evaluate_transaction_integration(self):
        """Test full transaction evaluation integration."""
        # Create a rule that should trigger
        rule = AlertRule.objects.create(
            name='Integration Test Rule',
            fraud_score_threshold=0.5,
            conditions={
                'fraud_check': {
                    'type': 'threshold',
                    'field': 'fraud_score',
                    'operator': '>',
                    'threshold': 0.5
                }
            },
            alert_type='integration_test',
            severity='high',
            action='alert',
            owner=self.user
        )
        
        alerts = self.alert_generator.evaluate_transaction(
            self.transaction, self.fraud_result, self.user
        )
        
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].alert_type, 'integration_test')
        self.assertEqual(alerts[0].severity, 'high')
        self.assertEqual(alerts[0].fraud_score, 0.75)
    
    def test_performance_metrics(self):
        """Test performance metrics tracking."""
        initial_metrics = self.alert_generator.get_performance_metrics()
        
        # Create a rule and evaluate transaction
        rule = AlertRule.objects.create(
            name='Metrics Test Rule',
            fraud_score_threshold=0.5,
            conditions={
                'fraud_check': {
                    'type': 'threshold',
                    'field': 'fraud_score',
                    'operator': '>',
                    'threshold': 0.5
                }
            },
            alert_type='metrics_test',
            severity='medium',
            action='alert',
            owner=self.user
        )
        
        self.alert_generator.evaluate_transaction(
            self.transaction, self.fraud_result, self.user
        )
        
        updated_metrics = self.alert_generator.get_performance_metrics()
        
        self.assertGreater(updated_metrics['alerts_generated'], initial_metrics['alerts_generated'])
        self.assertGreater(updated_metrics['rules_evaluated'], initial_metrics['rules_evaluated'])
        self.assertGreaterEqual(updated_metrics['avg_generation_time_ms'], 0)