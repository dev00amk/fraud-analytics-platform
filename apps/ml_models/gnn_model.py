"""
Graph Neural Network for relationship-driven fraud detection.
Production-ready GNN implementation with PyTorch Geometric.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool

logger = logging.getLogger(__name__)


@dataclass
class GraphFeatures:
    """Graph features for fraud detection."""

    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    batch: Optional[torch.Tensor] = None


class FraudGNN(nn.Module):
    """
    Graph Neural Network for fraud detection using transaction relationships.

    Architecture:
    - Graph Attention Network (GAT) layers for relationship modeling
    - Graph Convolutional Network (GCN) for feature propagation
    - Global pooling for graph-level predictions
    """

    def __init__(
        self,
        node_features: int = 64,
        edge_features: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        num_classes: int = 2,
    ):
        super(FraudGNN, self).__init__()

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Node feature projection
        self.node_proj = nn.Linear(node_features, hidden_dim)

        # GAT layers for attention-based aggregation
        self.gat_layers = nn.ModuleList(
            [
                GATConv(
                    hidden_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=edge_features,
                )
                for _ in range(num_layers)
            ]
        )

        # GCN layers for feature propagation
        self.gcn_layers = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Risk scoring head
        self.risk_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the GNN.

        Args:
            data: PyTorch Geometric Data object with node features, edge index, and edge features

        Returns:
            Dictionary with fraud predictions and risk scores
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = getattr(data, "batch", None)

        # Project node features
        x = self.node_proj(x)
        x = F.relu(x)

        # Store initial features for residual connections
        residual = x

        # GAT layers with residual connections
        for i, gat_layer in enumerate(self.gat_layers):
            x_gat = gat_layer(x, edge_index, edge_attr)
            x_gat = F.relu(x_gat)
            x_gat = F.dropout(x_gat, p=self.dropout, training=self.training)

            # Residual connection
            if i > 0:
                x_gat = x_gat + residual
            residual = x_gat
            x = x_gat

        # GCN layers for additional feature propagation
        for gcn_layer in self.gcn_layers:
            x_gcn = gcn_layer(x, edge_index)
            x_gcn = F.relu(x_gcn)
            x_gcn = F.dropout(x_gcn, p=self.dropout, training=self.training)
            x = x + x_gcn  # Residual connection

        # Global pooling for graph-level representation
        if batch is not None:
            # Batch processing
            graph_repr = global_mean_pool(x, batch)
            node_repr = x
        else:
            # Single graph
            graph_repr = torch.mean(x, dim=0, keepdim=True)
            node_repr = x

        # Combine graph and node representations
        combined_repr = torch.cat(
            [
                graph_repr,
                (
                    torch.mean(node_repr, dim=0, keepdim=True)
                    if batch is None
                    else global_mean_pool(node_repr, batch)
                ),
            ],
            dim=1,
        )

        # Predictions
        fraud_logits = self.classifier(combined_repr)
        risk_scores = self.risk_scorer(combined_repr)

        return {
            "fraud_logits": fraud_logits,
            "fraud_probs": F.softmax(fraud_logits, dim=1),
            "risk_scores": risk_scores,
            "node_embeddings": x,
            "graph_embeddings": combined_repr,
        }


class GraphBuilder:
    """
    Builds transaction graphs for GNN processing.
    Creates nodes for users, merchants, devices, and edges for relationships.
    """

    def __init__(self):
        self.node_types = {"user": 0, "merchant": 1, "device": 2, "ip": 3}

    def build_transaction_graph(
        self, transactions: List[Dict], time_window: timedelta = timedelta(hours=24)
    ) -> Data:
        """
        Build a graph from transaction data.

        Args:
            transactions: List of transaction dictionaries
            time_window: Time window for connecting transactions

        Returns:
            PyTorch Geometric Data object
        """
        nodes = {}
        edges = []
        node_features = []
        edge_features = []

        # Create nodes
        node_id = 0
        for txn in transactions:
            # User node
            user_key = f"user_{txn['user_id']}"
            if user_key not in nodes:
                nodes[user_key] = node_id
                node_features.append(self._get_user_features(txn))
                node_id += 1

            # Merchant node
            merchant_key = f"merchant_{txn['merchant_id']}"
            if merchant_key not in nodes:
                nodes[merchant_key] = node_id
                node_features.append(self._get_merchant_features(txn))
                node_id += 1

            # Device node (if available)
            if txn.get("device_fingerprint"):
                device_key = f"device_{txn['device_fingerprint']}"
                if device_key not in nodes:
                    nodes[device_key] = node_id
                    node_features.append(self._get_device_features(txn))
                    node_id += 1

            # IP node
            ip_key = f"ip_{txn['ip_address']}"
            if ip_key not in nodes:
                nodes[ip_key] = node_id
                node_features.append(self._get_ip_features(txn))
                node_id += 1

        # Create edges
        for i, txn1 in enumerate(transactions):
            for j, txn2 in enumerate(transactions[i + 1 :], i + 1):
                if self._should_connect(txn1, txn2, time_window):
                    edge_feat = self._get_edge_features(txn1, txn2)

                    # User-Merchant edge
                    user_node = nodes[f"user_{txn1['user_id']}"]
                    merchant_node = nodes[f"merchant_{txn1['merchant_id']}"]
                    edges.append([user_node, merchant_node])
                    edge_features.append(edge_feat)

                    # Bidirectional edge
                    edges.append([merchant_node, user_node])
                    edge_features.append(edge_feat)

        # Convert to tensors
        node_features = torch.tensor(node_features, dtype=torch.float32)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_features = torch.tensor(edge_features, dtype=torch.float32)

        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)

    def _get_user_features(self, txn: Dict) -> List[float]:
        """Extract user node features."""
        return [
            self.node_types["user"],  # Node type
            float(txn.get("amount", 0)),
            float(txn.get("user_transaction_count", 0)),
            float(txn.get("user_avg_amount", 0)),
            float(txn.get("user_velocity", 0)),
            # Add more user features...
        ] + [0.0] * (
            64 - 5
        )  # Pad to 64 features

    def _get_merchant_features(self, txn: Dict) -> List[float]:
        """Extract merchant node features."""
        return [
            self.node_types["merchant"],  # Node type
            float(txn.get("amount", 0)),
            float(txn.get("merchant_risk_score", 0)),
            float(txn.get("merchant_transaction_count", 0)),
            float(txn.get("merchant_avg_amount", 0)),
            # Add more merchant features...
        ] + [0.0] * (
            64 - 5
        )  # Pad to 64 features

    def _get_device_features(self, txn: Dict) -> List[float]:
        """Extract device node features."""
        return [
            self.node_types["device"],  # Node type
            float(txn.get("device_risk_score", 0)),
            float(txn.get("device_transaction_count", 0)),
            float(txn.get("device_unique_users", 0)),
            # Add more device features...
        ] + [0.0] * (
            64 - 4
        )  # Pad to 64 features

    def _get_ip_features(self, txn: Dict) -> List[float]:
        """Extract IP node features."""
        return [
            self.node_types["ip"],  # Node type
            float(txn.get("ip_risk_score", 0)),
            float(txn.get("ip_transaction_count", 0)),
            float(txn.get("ip_unique_users", 0)),
            # Add more IP features...
        ] + [0.0] * (
            64 - 4
        )  # Pad to 64 features

    def _get_edge_features(self, txn1: Dict, txn2: Dict) -> List[float]:
        """Extract edge features between transactions."""
        time_diff = abs(
            datetime.fromisoformat(txn1["timestamp"].replace("Z", "+00:00"))
            - datetime.fromisoformat(txn2["timestamp"].replace("Z", "+00:00"))
        ).total_seconds()

        amount_diff = abs(float(txn1["amount"]) - float(txn2["amount"]))

        return [
            time_diff / 3600,  # Time difference in hours
            amount_diff,
            float(txn1["user_id"] == txn2["user_id"]),  # Same user
            float(txn1["merchant_id"] == txn2["merchant_id"]),  # Same merchant
            float(txn1.get("ip_address") == txn2.get("ip_address")),  # Same IP
            # Add more edge features...
        ] + [0.0] * (
            16 - 5
        )  # Pad to 16 features

    def _should_connect(self, txn1: Dict, txn2: Dict, time_window: timedelta) -> bool:
        """Determine if two transactions should be connected."""
        time_diff = abs(
            datetime.fromisoformat(txn1["timestamp"].replace("Z", "+00:00"))
            - datetime.fromisoformat(txn2["timestamp"].replace("Z", "+00:00"))
        )

        # Connect if within time window and share common attributes
        if time_diff <= time_window:
            return (
                txn1["user_id"] == txn2["user_id"]
                or txn1["merchant_id"] == txn2["merchant_id"]
                or txn1.get("ip_address") == txn2.get("ip_address")
                or txn1.get("device_fingerprint") == txn2.get("device_fingerprint")
            )

        return False


class GNNInferenceEngine:
    """
    Production inference engine for GNN fraud detection.
    Handles model loading, caching, and real-time predictions.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.graph_builder = GraphBuilder()
        self.model.eval()

    def _load_model(self, model_path: str) -> FraudGNN:
        """Load trained GNN model."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            model = FraudGNN(**checkpoint["model_config"])
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device)
            logger.info(f"Loaded GNN model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load GNN model: {e}")
            # Return default model for fallback
            return FraudGNN().to(self.device)

    @torch.no_grad()
    def predict(
        self, transactions: List[Dict], return_embeddings: bool = False
    ) -> Dict[str, any]:
        """
        Make fraud predictions using GNN.

        Args:
            transactions: List of transaction data
            return_embeddings: Whether to return node embeddings

        Returns:
            Prediction results with fraud probabilities and risk scores
        """
        try:
            # Build graph
            graph_data = self.graph_builder.build_transaction_graph(transactions)
            graph_data = graph_data.to(self.device)

            # Model inference
            start_time = datetime.now()
            outputs = self.model(graph_data)
            inference_time = (datetime.now() - start_time).total_seconds() * 1000

            # Extract predictions
            fraud_probs = outputs["fraud_probs"].cpu().numpy()
            risk_scores = outputs["risk_scores"].cpu().numpy()

            results = {
                "fraud_probability": float(fraud_probs[0][1]),  # Probability of fraud
                "risk_score": float(risk_scores[0][0]),
                "inference_time_ms": inference_time,
                "model_type": "gnn",
                "confidence": float(max(fraud_probs[0])),
            }

            if return_embeddings:
                results["node_embeddings"] = (
                    outputs["node_embeddings"].cpu().numpy().tolist()
                )
                results["graph_embeddings"] = (
                    outputs["graph_embeddings"].cpu().numpy().tolist()
                )

            return results

        except Exception as e:
            logger.error(f"GNN prediction failed: {e}")
            return {
                "fraud_probability": 0.5,
                "risk_score": 0.5,
                "inference_time_ms": 0,
                "model_type": "gnn",
                "error": str(e),
            }
