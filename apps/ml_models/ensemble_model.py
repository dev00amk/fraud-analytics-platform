"""
Ensemble model combining XGBoost + LSTM + GNN for fraud detection.
Production-ready ensemble with weighted voting and confidence scoring.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .gnn_model import GNNInferenceEngine
from .transformer_model import TransformerInferenceEngine

logger = logging.getLogger(__name__)


@dataclass
class EnsemblePrediction:
    """Ensemble prediction result."""

    fraud_probability: float
    risk_score: float
    confidence: float
    model_predictions: Dict[str, Dict]
    ensemble_weights: Dict[str, float]
    inference_time_ms: float


class LSTMModel(nn.Module):
    """
    LSTM model for sequential fraud detection.
    Lightweight alternative to transformer for faster inference.
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        super(LSTMModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Output dimension
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

        # Risk scoring head
        self.risk_scorer = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through LSTM.

        Args:
            x: Input sequences [batch_size, seq_len, input_dim]

        Returns:
            Dictionary with predictions
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use last output for classification
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            last_output = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            last_output = hidden[-1]

        # Predictions
        fraud_logits = self.classifier(last_output)
        risk_scores = self.risk_scorer(last_output)

        return {
            "fraud_logits": fraud_logits,
            "fraud_probs": torch.softmax(fraud_logits, dim=1),
            "risk_scores": risk_scores,
            "lstm_output": lstm_out,
            "hidden_state": last_output,
        }


class XGBoostModel:
    """
    XGBoost model for tabular fraud detection.
    Handles feature engineering and gradient boosting.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load trained XGBoost model."""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.feature_names = model_data["feature_names"]
            logger.info(f"Loaded XGBoost model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {e}")
            # Initialize default model
            self.model = xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
            )

    def extract_features(self, transaction: Dict) -> np.ndarray:
        """Extract tabular features from transaction."""
        features = []

        # Basic transaction features
        features.extend(
            [
                float(transaction.get("amount", 0)),
                np.log1p(float(transaction.get("amount", 0))),  # Log amount
                float(transaction.get("amount", 0)) ** 0.5,  # Sqrt amount
            ]
        )

        # Categorical features (one-hot encoded)
        payment_methods = [
            "credit_card",
            "debit_card",
            "bank_transfer",
            "digital_wallet",
        ]
        for method in payment_methods:
            features.append(float(transaction.get("payment_method") == method))

        # Temporal features
        try:
            dt = datetime.fromisoformat(transaction["timestamp"].replace("Z", "+00:00"))
            features.extend(
                [
                    dt.hour,
                    dt.weekday(),
                    dt.day,
                    dt.month,
                    float(dt.weekday() >= 5),  # Is weekend
                    float(6 <= dt.hour <= 22),  # Is business hours
                ]
            )
        except:
            features.extend([0] * 6)

        # User behavior features
        features.extend(
            [
                float(transaction.get("user_transaction_count", 0)),
                float(transaction.get("user_daily_amount", 0)),
                float(transaction.get("user_avg_amount", 0)),
                float(transaction.get("user_velocity_1h", 0)),
                float(transaction.get("user_velocity_24h", 0)),
            ]
        )

        # Merchant features
        features.extend(
            [
                float(transaction.get("merchant_risk_score", 0.5)),
                float(transaction.get("merchant_transaction_count", 0)),
                float(transaction.get("merchant_avg_amount", 0)),
            ]
        )

        # Device and location features
        features.extend(
            [
                float(transaction.get("device_risk_score", 0.5)),
                float(transaction.get("ip_risk_score", 0.5)),
                float(transaction.get("geolocation_risk", 0.5)),
                float(bool(transaction.get("device_fingerprint"))),
            ]
        )

        # Derived features
        user_avg = float(transaction.get("user_avg_amount", 1))
        merchant_avg = float(transaction.get("merchant_avg_amount", 1))
        amount = float(transaction.get("amount", 0))

        features.extend(
            [
                amount / max(user_avg, 1),  # Amount vs user average
                amount / max(merchant_avg, 1),  # Amount vs merchant average
                float(amount > user_avg * 3),  # Large amount flag
                float(amount < user_avg * 0.1),  # Small amount flag
            ]
        )

        return np.array(features).reshape(1, -1)

    def predict(self, transaction: Dict) -> Dict[str, float]:
        """Make prediction using XGBoost."""
        try:
            # Extract features
            features = self.extract_features(transaction)

            # Scale features
            features_scaled = self.scaler.transform(features)

            # Predict
            if self.model:
                fraud_prob = self.model.predict_proba(features_scaled)[0][1]
                risk_score = fraud_prob  # Use probability as risk score
            else:
                fraud_prob = 0.5
                risk_score = 0.5

            return {
                "fraud_probability": float(fraud_prob),
                "risk_score": float(risk_score),
                "model_type": "xgboost",
            }

        except Exception as e:
            logger.error(f"XGBoost prediction failed: {e}")
            return {
                "fraud_probability": 0.5,
                "risk_score": 0.5,
                "model_type": "xgboost",
                "error": str(e),
            }


class EnsembleModel:
    """
    Ensemble model combining XGBoost, LSTM, and GNN predictions.
    Uses weighted voting with dynamic weight adjustment based on confidence.
    """

    def __init__(
        self,
        xgboost_path: Optional[str] = None,
        lstm_path: Optional[str] = None,
        gnn_path: Optional[str] = None,
        transformer_path: Optional[str] = None,
        weights_config: Optional[Dict[str, float]] = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)

        # Initialize models
        self.xgboost_model = XGBoostModel(xgboost_path)
        self.lstm_model = self._load_lstm_model(lstm_path) if lstm_path else None
        self.gnn_engine = GNNInferenceEngine(gnn_path, device) if gnn_path else None
        self.transformer_engine = (
            TransformerInferenceEngine(transformer_path, device)
            if transformer_path
            else None
        )

        # Default weights
        self.base_weights = weights_config or {
            "xgboost": 0.3,
            "lstm": 0.2,
            "gnn": 0.25,
            "transformer": 0.25,
        }

        # Performance tracking for dynamic weighting
        self.model_performance = {
            "xgboost": {"accuracy": 0.85, "confidence": 0.8},
            "lstm": {"accuracy": 0.82, "confidence": 0.75},
            "gnn": {"accuracy": 0.88, "confidence": 0.85},
            "transformer": {"accuracy": 0.90, "confidence": 0.87},
        }

    def _load_lstm_model(self, model_path: str) -> Optional[LSTMModel]:
        """Load LSTM model."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            model = LSTMModel(**checkpoint["model_config"])
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device)
            model.eval()
            logger.info(f"Loaded LSTM model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")
            return None

    def predict(
        self,
        transaction: Dict,
        user_transactions: Optional[Dict[str, List[Dict]]] = None,
        transaction_graph: Optional[List[Dict]] = None,
    ) -> EnsemblePrediction:
        """
        Make ensemble prediction combining all models.

        Args:
            transaction: Target transaction
            user_transactions: Historical user transactions for sequential models
            transaction_graph: Transaction graph data for GNN

        Returns:
            EnsemblePrediction with combined results
        """
        start_time = datetime.now()
        model_predictions = {}

        # XGBoost prediction (always available)
        xgb_pred = self.xgboost_model.predict(transaction)
        model_predictions["xgboost"] = xgb_pred

        # LSTM prediction
        if self.lstm_model and user_transactions:
            lstm_pred = self._predict_lstm(transaction, user_transactions)
            model_predictions["lstm"] = lstm_pred

        # GNN prediction
        if self.gnn_engine and transaction_graph:
            gnn_pred = self.gnn_engine.predict(transaction_graph)
            model_predictions["gnn"] = gnn_pred

        # Transformer prediction
        if self.transformer_engine and user_transactions:
            transformer_pred = self.transformer_engine.predict(
                transaction, user_transactions
            )
            model_predictions["transformer"] = transformer_pred

        # Calculate ensemble prediction
        ensemble_result = self._combine_predictions(model_predictions)

        # Calculate total inference time
        total_time = (datetime.now() - start_time).total_seconds() * 1000

        return EnsemblePrediction(
            fraud_probability=ensemble_result["fraud_probability"],
            risk_score=ensemble_result["risk_score"],
            confidence=ensemble_result["confidence"],
            model_predictions=model_predictions,
            ensemble_weights=ensemble_result["weights"],
            inference_time_ms=total_time,
        )

    def _predict_lstm(
        self, transaction: Dict, user_transactions: Dict[str, List[Dict]]
    ) -> Dict[str, float]:
        """Make LSTM prediction."""
        try:
            # Build sequence (simplified version)
            user_id = transaction["user_id"]
            user_txns = user_transactions.get(user_id, [])

            # Create sequence features (simplified)
            sequence_data = []
            for txn in user_txns[-50:]:  # Last 50 transactions
                features = self._extract_lstm_features(txn)
                sequence_data.append(features)

            # Add current transaction
            sequence_data.append(self._extract_lstm_features(transaction))

            # Pad sequence
            max_len = 50
            if len(sequence_data) < max_len:
                padding = [[0.0] * 64] * (max_len - len(sequence_data))
                sequence_data = padding + sequence_data
            else:
                sequence_data = sequence_data[-max_len:]

            # Convert to tensor
            sequence_tensor = torch.tensor([sequence_data], dtype=torch.float32).to(
                self.device
            )

            # Predict
            with torch.no_grad():
                outputs = self.lstm_model(sequence_tensor)
                fraud_prob = outputs["fraud_probs"][0][1].item()
                risk_score = outputs["risk_scores"][0][0].item()

            return {
                "fraud_probability": fraud_prob,
                "risk_score": risk_score,
                "model_type": "lstm",
            }

        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            return {
                "fraud_probability": 0.5,
                "risk_score": 0.5,
                "model_type": "lstm",
                "error": str(e),
            }

    def _extract_lstm_features(self, transaction: Dict) -> List[float]:
        """Extract features for LSTM model."""
        features = [
            float(transaction.get("amount", 0)) / 1000.0,  # Normalized amount
            float(transaction.get("user_transaction_count", 0)) / 100.0,
            float(transaction.get("merchant_risk_score", 0.5)),
            float(transaction.get("device_risk_score", 0.5)),
        ]

        # Pad to 64 features
        features.extend([0.0] * (64 - len(features)))
        return features

    def _combine_predictions(
        self, model_predictions: Dict[str, Dict]
    ) -> Dict[str, float]:
        """Combine predictions from multiple models using weighted voting."""
        if not model_predictions:
            return {
                "fraud_probability": 0.5,
                "risk_score": 0.5,
                "confidence": 0.0,
                "weights": {},
            }

        # Calculate dynamic weights based on model availability and performance
        available_models = list(model_predictions.keys())
        weights = self._calculate_dynamic_weights(available_models, model_predictions)

        # Weighted average of predictions
        weighted_fraud_prob = 0.0
        weighted_risk_score = 0.0
        total_weight = 0.0

        for model_name, prediction in model_predictions.items():
            if "error" not in prediction:
                weight = weights.get(model_name, 0.0)
                weighted_fraud_prob += prediction["fraud_probability"] * weight
                weighted_risk_score += prediction["risk_score"] * weight
                total_weight += weight

        # Normalize if we have valid predictions
        if total_weight > 0:
            weighted_fraud_prob /= total_weight
            weighted_risk_score /= total_weight
        else:
            weighted_fraud_prob = 0.5
            weighted_risk_score = 0.5

        # Calculate ensemble confidence
        confidence = self._calculate_ensemble_confidence(model_predictions, weights)

        return {
            "fraud_probability": weighted_fraud_prob,
            "risk_score": weighted_risk_score,
            "confidence": confidence,
            "weights": weights,
        }

    def _calculate_dynamic_weights(
        self, available_models: List[str], model_predictions: Dict[str, Dict]
    ) -> Dict[str, float]:
        """Calculate dynamic weights based on model performance and confidence."""
        weights = {}

        for model_name in available_models:
            if "error" in model_predictions[model_name]:
                weights[model_name] = 0.0
                continue

            # Base weight
            base_weight = self.base_weights.get(model_name, 0.25)

            # Performance adjustment
            performance = self.model_performance.get(
                model_name, {"accuracy": 0.8, "confidence": 0.8}
            )
            performance_factor = (
                performance["accuracy"] + performance["confidence"]
            ) / 2

            # Prediction confidence adjustment
            pred_confidence = model_predictions[model_name].get("confidence", 0.5)
            confidence_factor = pred_confidence

            # Final weight
            weights[model_name] = base_weight * performance_factor * confidence_factor

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        return weights

    def _calculate_ensemble_confidence(
        self, model_predictions: Dict[str, Dict], weights: Dict[str, float]
    ) -> float:
        """Calculate overall ensemble confidence."""
        if not model_predictions:
            return 0.0

        # Agreement-based confidence
        fraud_probs = [
            pred["fraud_probability"]
            for pred in model_predictions.values()
            if "error" not in pred
        ]

        if len(fraud_probs) < 2:
            return 0.5

        # Calculate variance (lower variance = higher confidence)
        variance = np.var(fraud_probs)
        agreement_confidence = 1.0 / (1.0 + variance * 10)  # Scale variance

        # Weighted confidence from individual models
        weighted_confidence = sum(
            pred.get("confidence", 0.5) * weights.get(model_name, 0.0)
            for model_name, pred in model_predictions.items()
            if "error" not in pred
        )

        # Combine agreement and individual confidences
        ensemble_confidence = (agreement_confidence + weighted_confidence) / 2

        return min(max(ensemble_confidence, 0.0), 1.0)

    def update_model_performance(
        self, model_name: str, accuracy: float, confidence: float
    ):
        """Update model performance metrics for dynamic weighting."""
        if model_name in self.model_performance:
            # Exponential moving average
            alpha = 0.1
            self.model_performance[model_name]["accuracy"] = (
                alpha * accuracy
                + (1 - alpha) * self.model_performance[model_name]["accuracy"]
            )
            self.model_performance[model_name]["confidence"] = (
                alpha * confidence
                + (1 - alpha) * self.model_performance[model_name]["confidence"]
            )


class EnsembleInferenceEngine:
    """
    Production inference engine for ensemble fraud detection.
    Handles model orchestration, caching, and performance monitoring.
    """

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.ensemble = EnsembleModel(**self.config["model_paths"])

        # Performance monitoring
        self.prediction_cache = {}
        self.performance_metrics = {
            "total_predictions": 0,
            "avg_inference_time": 0.0,
            "cache_hit_rate": 0.0,
        }

    def _load_config(self, config_path: str) -> Dict:
        """Load ensemble configuration."""
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {"model_paths": {}, "cache_ttl": 300, "performance_tracking": True}

    def predict(
        self,
        transaction: Dict,
        user_transactions: Optional[Dict[str, List[Dict]]] = None,
        transaction_graph: Optional[List[Dict]] = None,
        use_cache: bool = True,
    ) -> EnsemblePrediction:
        """
        Make ensemble prediction with caching and monitoring.

        Args:
            transaction: Target transaction
            user_transactions: Historical user transactions
            transaction_graph: Transaction graph data
            use_cache: Whether to use prediction caching

        Returns:
            EnsemblePrediction result
        """
        # Generate cache key
        cache_key = self._generate_cache_key(transaction)

        # Check cache
        if use_cache and cache_key in self.prediction_cache:
            cached_result = self.prediction_cache[cache_key]
            if self._is_cache_valid(cached_result):
                self.performance_metrics["cache_hit_rate"] += 1
                return cached_result["prediction"]

        # Make prediction
        prediction = self.ensemble.predict(
            transaction, user_transactions, transaction_graph
        )

        # Cache result
        if use_cache:
            self.prediction_cache[cache_key] = {
                "prediction": prediction,
                "timestamp": datetime.now(),
            }

        # Update metrics
        self._update_metrics(prediction)

        return prediction

    def _generate_cache_key(self, transaction: Dict) -> str:
        """Generate cache key for transaction."""
        key_data = {
            "transaction_id": transaction.get("transaction_id"),
            "user_id": transaction.get("user_id"),
            "amount": transaction.get("amount"),
            "timestamp": transaction.get("timestamp"),
        }
        return str(hash(str(sorted(key_data.items()))))

    def _is_cache_valid(self, cached_result: Dict) -> bool:
        """Check if cached result is still valid."""
        cache_ttl = self.config.get("cache_ttl", 300)  # 5 minutes default
        age = (datetime.now() - cached_result["timestamp"]).total_seconds()
        return age < cache_ttl

    def _update_metrics(self, prediction: EnsemblePrediction):
        """Update performance metrics."""
        self.performance_metrics["total_predictions"] += 1

        # Update average inference time
        alpha = 0.1
        self.performance_metrics["avg_inference_time"] = (
            alpha * prediction.inference_time_ms
            + (1 - alpha) * self.performance_metrics["avg_inference_time"]
        )

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        total_requests = self.performance_metrics["total_predictions"]
        cache_hits = self.performance_metrics["cache_hit_rate"]

        return {
            "total_predictions": total_requests,
            "avg_inference_time_ms": self.performance_metrics["avg_inference_time"],
            "cache_hit_rate": cache_hits / max(total_requests, 1),
            "predictions_per_second": 1000
            / max(self.performance_metrics["avg_inference_time"], 1),
        }
