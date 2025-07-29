"""
Transformer-based sequential model for transaction pattern analysis.
Production-ready implementation with attention mechanisms for fraud detection.
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class TransactionSequence:
    """Transaction sequence data structure."""

    sequences: torch.Tensor  # [batch_size, seq_len, features]
    attention_mask: torch.Tensor  # [batch_size, seq_len]
    timestamps: torch.Tensor  # [batch_size, seq_len]
    labels: Optional[torch.Tensor] = None  # [batch_size]


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer with temporal awareness.
    Combines sinusoidal encoding with learned temporal embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Standard sinusoidal positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

        # Learnable temporal embeddings for time-of-day, day-of-week patterns
        self.hour_embedding = nn.Embedding(24, d_model // 4)
        self.day_embedding = nn.Embedding(7, d_model // 4)
        self.temporal_proj = nn.Linear(d_model // 2, d_model)

    def forward(
        self, x: torch.Tensor, timestamps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input embeddings [seq_len, batch_size, d_model]
            timestamps: Unix timestamps [batch_size, seq_len]

        Returns:
            Position-encoded embeddings
        """
        seq_len = x.size(0)

        # Standard positional encoding
        pos_encoding = self.pe[:seq_len, :]

        # Temporal encoding if timestamps provided
        if timestamps is not None:
            # Convert timestamps to datetime features
            dt_features = self._extract_datetime_features(timestamps)
            hour_emb = self.hour_embedding(
                dt_features["hour"]
            )  # [batch_size, seq_len, d_model//4]
            day_emb = self.day_embedding(
                dt_features["day"]
            )  # [batch_size, seq_len, d_model//4]

            # Combine temporal embeddings
            temporal_emb = torch.cat(
                [hour_emb, day_emb], dim=-1
            )  # [batch_size, seq_len, d_model//2]
            temporal_emb = self.temporal_proj(
                temporal_emb
            )  # [batch_size, seq_len, d_model]
            temporal_emb = temporal_emb.transpose(
                0, 1
            )  # [seq_len, batch_size, d_model]

            # Combine positional and temporal encodings
            pos_encoding = pos_encoding + temporal_emb

        return self.dropout(x + pos_encoding)

    def _extract_datetime_features(
        self, timestamps: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Extract hour and day features from timestamps."""
        # Convert to datetime (assuming timestamps are in seconds)
        dt_array = timestamps.cpu().numpy()

        # Extract features
        hours = []
        days = []

        for batch in dt_array:
            batch_hours = []
            batch_days = []
            for ts in batch:
                dt = datetime.fromtimestamp(ts)
                batch_hours.append(dt.hour)
                batch_days.append(dt.weekday())
            hours.append(batch_hours)
            days.append(batch_days)

        return {
            "hour": torch.tensor(hours, dtype=torch.long, device=timestamps.device),
            "day": torch.tensor(days, dtype=torch.long, device=timestamps.device),
        }


class FraudTransformer(nn.Module):
    """
    Transformer model for sequential fraud detection.

    Architecture:
    - Input embedding with feature projection
    - Positional encoding with temporal awareness
    - Multi-head self-attention layers
    - Feed-forward networks with residual connections
    - Classification and risk scoring heads
    """

    def __init__(
        self,
        input_dim: int = 128,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 100,
        num_classes: int = 2,
    ):
        super(FraudTransformer, self).__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Attention pooling for sequence aggregation
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=False
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_classes),
        )

        # Risk scoring head
        self.risk_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        # Anomaly detection head
        self.anomaly_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.xavier_uniform_(module.out_proj.weight)

    def forward(
        self,
        sequences: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the transformer.

        Args:
            sequences: Input sequences [batch_size, seq_len, input_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            timestamps: Timestamps [batch_size, seq_len]

        Returns:
            Dictionary with predictions and attention weights
        """
        batch_size, seq_len, _ = sequences.shape

        # Input projection
        x = self.input_projection(sequences)  # [batch_size, seq_len, d_model]
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]

        # Positional encoding
        x = self.pos_encoder(x, timestamps)

        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert to transformer format (True = ignore)
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None

        # Transformer encoding
        encoded = self.transformer_encoder(
            x, src_key_padding_mask=src_key_padding_mask
        )  # [seq_len, batch_size, d_model]

        # Attention pooling for sequence representation
        query = encoded[-1:, :, :]  # Use last token as query
        pooled_output, attention_weights = self.attention_pool(
            query, encoded, encoded, key_padding_mask=src_key_padding_mask
        )
        pooled_output = pooled_output.squeeze(0)  # [batch_size, d_model]

        # Predictions
        fraud_logits = self.classifier(pooled_output)
        risk_scores = self.risk_scorer(pooled_output)
        anomaly_scores = self.anomaly_detector(pooled_output)

        return {
            "fraud_logits": fraud_logits,
            "fraud_probs": F.softmax(fraud_logits, dim=1),
            "risk_scores": risk_scores,
            "anomaly_scores": anomaly_scores,
            "attention_weights": attention_weights,
            "sequence_embeddings": encoded,
            "pooled_embeddings": pooled_output,
        }


class SequenceBuilder:
    """
    Builds transaction sequences for transformer processing.
    Handles sequence padding, masking, and feature engineering.
    """

    def __init__(self, max_seq_len: int = 100, feature_dim: int = 128):
        self.max_seq_len = max_seq_len
        self.feature_dim = feature_dim

    def build_sequences(
        self,
        user_transactions: Dict[str, List[Dict]],
        target_transaction: Dict,
        lookback_hours: int = 24,
    ) -> TransactionSequence:
        """
        Build transaction sequences for a user.

        Args:
            user_transactions: Historical transactions by user
            target_transaction: Current transaction to analyze
            lookback_hours: Hours to look back for sequence

        Returns:
            TransactionSequence object
        """
        user_id = target_transaction["user_id"]
        target_time = datetime.fromisoformat(
            target_transaction["timestamp"].replace("Z", "+00:00")
        )

        # Get user's historical transactions
        user_txns = user_transactions.get(user_id, [])

        # Filter transactions within lookback window
        cutoff_time = target_time - timedelta(hours=lookback_hours)
        recent_txns = [
            txn
            for txn in user_txns
            if datetime.fromisoformat(txn["timestamp"].replace("Z", "+00:00"))
            >= cutoff_time
            and datetime.fromisoformat(txn["timestamp"].replace("Z", "+00:00"))
            < target_time
        ]

        # Sort by timestamp
        recent_txns.sort(key=lambda x: x["timestamp"])

        # Add target transaction
        recent_txns.append(target_transaction)

        # Build sequence features
        sequence_features = []
        timestamps = []

        for txn in recent_txns[-self.max_seq_len :]:  # Keep only recent transactions
            features = self._extract_transaction_features(txn)
            sequence_features.append(features)
            timestamps.append(
                datetime.fromisoformat(
                    txn["timestamp"].replace("Z", "+00:00")
                ).timestamp()
            )

        # Pad sequences
        seq_len = len(sequence_features)
        if seq_len < self.max_seq_len:
            # Pad with zeros
            padding = [[0.0] * self.feature_dim] * (self.max_seq_len - seq_len)
            sequence_features = padding + sequence_features
            timestamps = [0.0] * (self.max_seq_len - seq_len) + timestamps

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [0] * (self.max_seq_len - seq_len) + [1] * seq_len

        # Convert to tensors
        sequences = torch.tensor([sequence_features], dtype=torch.float32)
        attention_mask = torch.tensor([attention_mask], dtype=torch.bool)
        timestamps = torch.tensor([timestamps], dtype=torch.float32)

        return TransactionSequence(
            sequences=sequences, attention_mask=attention_mask, timestamps=timestamps
        )

    def _extract_transaction_features(self, txn: Dict) -> List[float]:
        """Extract features from a single transaction."""
        # Basic transaction features
        features = [
            float(txn.get("amount", 0)),
            float(txn.get("amount", 0)) / 1000.0,  # Normalized amount
            float(len(txn.get("merchant_id", ""))),  # Merchant ID length
            float(txn.get("payment_method") == "credit_card"),
            float(txn.get("payment_method") == "debit_card"),
            float(txn.get("payment_method") == "bank_transfer"),
        ]

        # Temporal features
        dt = datetime.fromisoformat(txn["timestamp"].replace("Z", "+00:00"))
        features.extend(
            [
                float(dt.hour) / 24.0,  # Hour of day (normalized)
                float(dt.weekday()) / 7.0,  # Day of week (normalized)
                float(dt.day) / 31.0,  # Day of month (normalized)
                float(dt.month) / 12.0,  # Month (normalized)
            ]
        )

        # Velocity features (if available)
        features.extend(
            [
                float(txn.get("user_transaction_count", 0)) / 100.0,
                float(txn.get("user_daily_amount", 0)) / 10000.0,
                float(txn.get("user_velocity_1h", 0)),
                float(txn.get("user_velocity_24h", 0)),
            ]
        )

        # Risk features
        features.extend(
            [
                float(txn.get("merchant_risk_score", 0.5)),
                float(txn.get("ip_risk_score", 0.5)),
                float(txn.get("device_risk_score", 0.5)),
                float(txn.get("geolocation_risk", 0.5)),
            ]
        )

        # Behavioral features
        features.extend(
            [
                float(txn.get("is_weekend", 0)),
                float(txn.get("is_night_time", 0)),
                float(txn.get("amount_deviation", 0)),
                float(txn.get("merchant_frequency", 0)),
            ]
        )

        # Pad or truncate to feature_dim
        if len(features) < self.feature_dim:
            features.extend([0.0] * (self.feature_dim - len(features)))
        else:
            features = features[: self.feature_dim]

        return features


class TransformerInferenceEngine:
    """
    Production inference engine for transformer fraud detection.
    Handles model loading, sequence building, and real-time predictions.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.sequence_builder = SequenceBuilder()
        self.model.eval()

    def _load_model(self, model_path: str) -> FraudTransformer:
        """Load trained transformer model."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            model = FraudTransformer(**checkpoint["model_config"])
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device)
            logger.info(f"Loaded Transformer model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load Transformer model: {e}")
            # Return default model for fallback
            return FraudTransformer().to(self.device)

    @torch.no_grad()
    def predict(
        self,
        target_transaction: Dict,
        user_transactions: Dict[str, List[Dict]],
        return_attention: bool = False,
    ) -> Dict[str, any]:
        """
        Make fraud predictions using transformer.

        Args:
            target_transaction: Transaction to analyze
            user_transactions: Historical transactions by user
            return_attention: Whether to return attention weights

        Returns:
            Prediction results with fraud probabilities and risk scores
        """
        try:
            # Build sequence
            sequence_data = self.sequence_builder.build_sequences(
                user_transactions, target_transaction
            )

            # Move to device
            sequences = sequence_data.sequences.to(self.device)
            attention_mask = sequence_data.attention_mask.to(self.device)
            timestamps = sequence_data.timestamps.to(self.device)

            # Model inference
            start_time = datetime.now()
            outputs = self.model(sequences, attention_mask, timestamps)
            inference_time = (datetime.now() - start_time).total_seconds() * 1000

            # Extract predictions
            fraud_probs = outputs["fraud_probs"].cpu().numpy()
            risk_scores = outputs["risk_scores"].cpu().numpy()
            anomaly_scores = outputs["anomaly_scores"].cpu().numpy()

            results = {
                "fraud_probability": float(fraud_probs[0][1]),
                "risk_score": float(risk_scores[0][0]),
                "anomaly_score": float(anomaly_scores[0][0]),
                "inference_time_ms": inference_time,
                "model_type": "transformer",
                "confidence": float(max(fraud_probs[0])),
            }

            if return_attention:
                attention_weights = outputs["attention_weights"].cpu().numpy()
                results["attention_weights"] = attention_weights.tolist()

            return results

        except Exception as e:
            logger.error(f"Transformer prediction failed: {e}")
            return {
                "fraud_probability": 0.5,
                "risk_score": 0.5,
                "anomaly_score": 0.5,
                "inference_time_ms": 0,
                "model_type": "transformer",
                "error": str(e),
            }
