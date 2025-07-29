"""
Real-time feature engineering pipeline for fraud detection.
Production-ready feature extraction with behavioral, velocity, and device features.
"""

import asyncio
import hashlib
import json
import logging
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import redis

logger = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    """Container for extracted features."""

    transaction_features: Dict[str, float] = field(default_factory=dict)
    user_features: Dict[str, float] = field(default_factory=dict)
    merchant_features: Dict[str, float] = field(default_factory=dict)
    device_features: Dict[str, float] = field(default_factory=dict)
    behavioral_features: Dict[str, float] = field(default_factory=dict)
    velocity_features: Dict[str, float] = field(default_factory=dict)
    network_features: Dict[str, float] = field(default_factory=dict)
    temporal_features: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        """Convert all features to flat dictionary."""
        all_features = {}
        for feature_type in [
            "transaction_features",
            "user_features",
            "merchant_features",
            "device_features",
            "behavioral_features",
            "velocity_features",
            "network_features",
            "temporal_features",
        ]:
            features = getattr(self, feature_type)
            for key, value in features.items():
                all_features[f"{feature_type}_{key}"] = value
        return all_features

    def to_array(self, feature_names: List[str]) -> np.ndarray:
        """Convert to numpy array with specified feature order."""
        feature_dict = self.to_dict()
        return np.array([feature_dict.get(name, 0.0) for name in feature_names])


class VelocityTracker:
    """
    Tracks transaction velocity for users, merchants, and devices.
    Uses sliding window approach with Redis for persistence.
    """

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.local_cache = defaultdict(lambda: defaultdict(deque))
        self.time_windows = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "6h": 21600,
            "24h": 86400,
            "7d": 604800,
        }

    def update_velocity(
        self,
        entity_type: str,
        entity_id: str,
        transaction: Dict[str, Any],
        timestamp: Optional[datetime] = None,
    ):
        """Update velocity tracking for an entity."""
        if timestamp is None:
            timestamp = datetime.now()

        # Create transaction record
        txn_record = {
            "amount": float(transaction.get("amount", 0)),
            "timestamp": timestamp.timestamp(),
            "transaction_id": transaction.get("transaction_id", ""),
        }

        # Update local cache
        entity_key = f"{entity_type}:{entity_id}"
        self.local_cache[entity_key]["transactions"].append(txn_record)

        # Update Redis if available
        if self.redis_client:
            try:
                redis_key = f"velocity:{entity_key}"
                self.redis_client.lpush(redis_key, json.dumps(txn_record))
                self.redis_client.expire(redis_key, self.time_windows["7d"])
            except Exception as e:
                logger.warning(f"Redis velocity update failed: {e}")

    def get_velocity_features(
        self, entity_type: str, entity_id: str, current_time: Optional[datetime] = None
    ) -> Dict[str, float]:
        """Get velocity features for an entity."""
        if current_time is None:
            current_time = datetime.now()

        current_timestamp = current_time.timestamp()
        entity_key = f"{entity_type}:{entity_id}"

        # Get transactions from cache or Redis
        transactions = self._get_transactions(entity_key, current_timestamp)

        features = {}

        # Calculate velocity for each time window
        for window_name, window_seconds in self.time_windows.items():
            cutoff_time = current_timestamp - window_seconds

            # Filter transactions in window
            window_txns = [
                txn for txn in transactions if txn["timestamp"] >= cutoff_time
            ]

            # Calculate features
            features.update(
                {
                    f"count_{window_name}": len(window_txns),
                    f"amount_sum_{window_name}": sum(
                        txn["amount"] for txn in window_txns
                    ),
                    f"amount_avg_{window_name}": (
                        np.mean([txn["amount"] for txn in window_txns])
                        if window_txns
                        else 0.0
                    ),
                    f"amount_std_{window_name}": (
                        np.std([txn["amount"] for txn in window_txns])
                        if len(window_txns) > 1
                        else 0.0
                    ),
                    f"amount_max_{window_name}": (
                        max([txn["amount"] for txn in window_txns])
                        if window_txns
                        else 0.0
                    ),
                    f"amount_min_{window_name}": (
                        min([txn["amount"] for txn in window_txns])
                        if window_txns
                        else 0.0
                    ),
                }
            )

        # Calculate velocity ratios
        features.update(
            {
                "velocity_ratio_1m_1h": features["count_1m"]
                / max(features["count_1h"], 1),
                "velocity_ratio_5m_1h": features["count_5m"]
                / max(features["count_1h"], 1),
                "velocity_ratio_1h_24h": features["count_1h"]
                / max(features["count_24h"], 1),
                "amount_ratio_1m_1h": features["amount_sum_1m"]
                / max(features["amount_sum_1h"], 1),
                "amount_ratio_1h_24h": features["amount_sum_1h"]
                / max(features["amount_sum_24h"], 1),
            }
        )

        return features

    def _get_transactions(
        self, entity_key: str, current_timestamp: float
    ) -> List[Dict]:
        """Get transactions from cache or Redis."""
        transactions = []

        # Try Redis first
        if self.redis_client:
            try:
                redis_key = f"velocity:{entity_key}"
                txn_data = self.redis_client.lrange(redis_key, 0, -1)
                for data in txn_data:
                    txn = json.loads(data)
                    if current_timestamp - txn["timestamp"] <= self.time_windows["7d"]:
                        transactions.append(txn)
            except Exception as e:
                logger.warning(f"Redis velocity read failed: {e}")

        # Fallback to local cache
        if not transactions and entity_key in self.local_cache:
            cache_txns = list(self.local_cache[entity_key]["transactions"])
            transactions = [
                txn
                for txn in cache_txns
                if current_timestamp - txn["timestamp"] <= self.time_windows["7d"]
            ]

        return transactions


class BehavioralAnalyzer:
    """
    Analyzes user behavioral patterns for anomaly detection.
    Tracks spending patterns, merchant preferences, and temporal behavior.
    """

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.user_profiles = defaultdict(dict)

    def update_user_profile(self, user_id: str, transaction: Dict[str, Any]):
        """Update user behavioral profile."""
        profile_key = f"profile:{user_id}"

        # Extract behavioral signals
        signals = {
            "amount": float(transaction.get("amount", 0)),
            "merchant_id": transaction.get("merchant_id", ""),
            "payment_method": transaction.get("payment_method", ""),
            "timestamp": datetime.fromisoformat(
                transaction["timestamp"].replace("Z", "+00:00")
            ).timestamp(),
            "device_fingerprint": transaction.get("device_fingerprint", ""),
            "ip_address": transaction.get("ip_address", ""),
        }

        # Update local profile
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "transactions": deque(maxlen=1000),
                "merchants": defaultdict(int),
                "payment_methods": defaultdict(int),
                "devices": defaultdict(int),
                "ips": defaultdict(int),
                "amounts": deque(maxlen=100),
                "timestamps": deque(maxlen=100),
            }

        profile = self.user_profiles[user_id]
        profile["transactions"].append(signals)
        profile["merchants"][signals["merchant_id"]] += 1
        profile["payment_methods"][signals["payment_method"]] += 1
        profile["devices"][signals["device_fingerprint"]] += 1
        profile["ips"][signals["ip_address"]] += 1
        profile["amounts"].append(signals["amount"])
        profile["timestamps"].append(signals["timestamp"])

        # Update Redis profile
        if self.redis_client:
            try:
                redis_profile = {
                    "total_transactions": len(profile["transactions"]),
                    "avg_amount": (
                        np.mean(profile["amounts"]) if profile["amounts"] else 0.0
                    ),
                    "std_amount": (
                        np.std(profile["amounts"])
                        if len(profile["amounts"]) > 1
                        else 0.0
                    ),
                    "unique_merchants": len(profile["merchants"]),
                    "unique_devices": len(profile["devices"]),
                    "unique_ips": len(profile["ips"]),
                    "last_updated": datetime.now().timestamp(),
                }
                self.redis_client.hset(profile_key, mapping=redis_profile)
                self.redis_client.expire(profile_key, 86400 * 30)  # 30 days
            except Exception as e:
                logger.warning(f"Redis profile update failed: {e}")

    def get_behavioral_features(
        self, user_id: str, current_transaction: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract behavioral anomaly features."""
        profile = self.user_profiles.get(user_id, {})
        features = {}

        if not profile or not profile.get("transactions"):
            # New user - return default features
            return {
                "is_new_user": 1.0,
                "merchant_familiarity": 0.0,
                "payment_method_familiarity": 0.0,
                "device_familiarity": 0.0,
                "ip_familiarity": 0.0,
                "amount_deviation": 0.0,
                "time_deviation": 0.0,
                "spending_pattern_score": 0.5,
            }

        current_amount = float(current_transaction.get("amount", 0))
        current_merchant = current_transaction.get("merchant_id", "")
        current_payment = current_transaction.get("payment_method", "")
        current_device = current_transaction.get("device_fingerprint", "")
        current_ip = current_transaction.get("ip_address", "")
        current_time = datetime.fromisoformat(
            current_transaction["timestamp"].replace("Z", "+00:00")
        ).timestamp()

        # Merchant familiarity
        merchant_count = profile["merchants"].get(current_merchant, 0)
        total_merchants = len(profile["merchants"])
        features["merchant_familiarity"] = merchant_count / max(
            len(profile["transactions"]), 1
        )
        features["merchant_diversity"] = total_merchants / max(
            len(profile["transactions"]), 1
        )

        # Payment method familiarity
        payment_count = profile["payment_methods"].get(current_payment, 0)
        features["payment_method_familiarity"] = payment_count / max(
            len(profile["transactions"]), 1
        )

        # Device familiarity
        device_count = profile["devices"].get(current_device, 0)
        features["device_familiarity"] = device_count / max(
            len(profile["transactions"]), 1
        )
        features["device_diversity"] = len(profile["devices"]) / max(
            len(profile["transactions"]), 1
        )

        # IP familiarity
        ip_count = profile["ips"].get(current_ip, 0)
        features["ip_familiarity"] = ip_count / max(len(profile["transactions"]), 1)
        features["ip_diversity"] = len(profile["ips"]) / max(
            len(profile["transactions"]), 1
        )

        # Amount deviation
        if profile["amounts"]:
            avg_amount = np.mean(profile["amounts"])
            std_amount = (
                np.std(profile["amounts"])
                if len(profile["amounts"]) > 1
                else avg_amount * 0.1
            )
            features["amount_deviation"] = abs(current_amount - avg_amount) / max(
                std_amount, 1
            )
            features["amount_percentile"] = np.percentile(
                profile["amounts"], [25, 50, 75, 90, 95, 99]
            )
            features["amount_z_score"] = (current_amount - avg_amount) / max(
                std_amount, 1
            )
        else:
            features["amount_deviation"] = 0.0
            features["amount_z_score"] = 0.0

        # Temporal patterns
        if profile["timestamps"]:
            # Hour of day pattern
            hours = [datetime.fromtimestamp(ts).hour for ts in profile["timestamps"]]
            current_hour = datetime.fromtimestamp(current_time).hour
            hour_counts = np.bincount(hours, minlength=24)
            features["hour_familiarity"] = hour_counts[current_hour] / max(
                len(hours), 1
            )

            # Day of week pattern
            days = [
                datetime.fromtimestamp(ts).weekday() for ts in profile["timestamps"]
            ]
            current_day = datetime.fromtimestamp(current_time).weekday()
            day_counts = np.bincount(days, minlength=7)
            features["day_familiarity"] = day_counts[current_day] / max(len(days), 1)

            # Time since last transaction
            last_timestamp = max(profile["timestamps"])
            features["time_since_last"] = (
                current_time - last_timestamp
            ) / 3600  # Hours
        else:
            features["hour_familiarity"] = 0.0
            features["day_familiarity"] = 0.0
            features["time_since_last"] = 0.0

        # Composite scores
        features["overall_familiarity"] = np.mean(
            [
                features["merchant_familiarity"],
                features["payment_method_familiarity"],
                features["device_familiarity"],
                features["ip_familiarity"],
            ]
        )

        features["anomaly_score"] = np.mean(
            [
                min(features["amount_deviation"], 5.0) / 5.0,  # Cap at 5 std devs
                1.0 - features["overall_familiarity"],
                1.0 - features["hour_familiarity"],
                1.0 - features["day_familiarity"],
            ]
        )

        return features


class DeviceFingerprinter:
    """
    Analyzes device fingerprints and generates device-based features.
    Tracks device reputation and behavioral patterns.
    """

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.device_profiles = defaultdict(dict)

    def analyze_device(self, transaction: Dict[str, Any]) -> Dict[str, float]:
        """Analyze device fingerprint and generate features."""
        device_fp = transaction.get("device_fingerprint", "")
        user_agent = transaction.get("user_agent", "")
        ip_address = transaction.get("ip_address", "")

        features = {}

        # Device fingerprint analysis
        if device_fp:
            features.update(self._analyze_fingerprint(device_fp))
            features.update(self._get_device_reputation(device_fp))
        else:
            features.update(
                {
                    "has_device_fingerprint": 0.0,
                    "device_entropy": 0.0,
                    "device_reputation": 0.5,
                    "device_age_days": 0.0,
                    "device_user_count": 0.0,
                }
            )

        # User agent analysis
        if user_agent:
            features.update(self._analyze_user_agent(user_agent))
        else:
            features.update(
                {
                    "has_user_agent": 0.0,
                    "browser_risk": 0.5,
                    "os_risk": 0.5,
                    "is_mobile": 0.0,
                    "is_bot": 0.0,
                }
            )

        # IP analysis
        if ip_address:
            features.update(self._analyze_ip_address(ip_address))
        else:
            features.update(
                {
                    "ip_reputation": 0.5,
                    "is_tor": 0.0,
                    "is_vpn": 0.0,
                    "is_datacenter": 0.0,
                    "geolocation_risk": 0.5,
                }
            )

        return features

    def _analyze_fingerprint(self, device_fp: str) -> Dict[str, float]:
        """Analyze device fingerprint characteristics."""
        features = {
            "has_device_fingerprint": 1.0,
            "fingerprint_length": len(device_fp),
            "fingerprint_entropy": self._calculate_entropy(device_fp),
        }

        # Check for suspicious patterns
        features["has_suspicious_chars"] = float(
            any(char in device_fp for char in ["<", ">", "script", "null", "undefined"])
        )

        # Fingerprint complexity
        unique_chars = len(set(device_fp))
        features["fingerprint_complexity"] = unique_chars / max(len(device_fp), 1)

        return features

    def _get_device_reputation(self, device_fp: str) -> Dict[str, float]:
        """Get device reputation from historical data."""
        device_key = f"device:{hashlib.md5(device_fp.encode()).hexdigest()}"

        # Try to get from Redis
        if self.redis_client:
            try:
                device_data = self.redis_client.hgetall(device_key)
                if device_data:
                    return {
                        "device_reputation": float(device_data.get(b"reputation", 0.5)),
                        "device_age_days": float(device_data.get(b"age_days", 0)),
                        "device_user_count": float(device_data.get(b"user_count", 0)),
                        "device_transaction_count": float(
                            device_data.get(b"txn_count", 0)
                        ),
                        "device_fraud_rate": float(device_data.get(b"fraud_rate", 0)),
                    }
            except Exception as e:
                logger.warning(f"Redis device lookup failed: {e}")

        # Default values for new devices
        return {
            "device_reputation": 0.5,
            "device_age_days": 0.0,
            "device_user_count": 0.0,
            "device_transaction_count": 0.0,
            "device_fraud_rate": 0.0,
        }

    def _analyze_user_agent(self, user_agent: str) -> Dict[str, float]:
        """Analyze user agent string."""
        ua_lower = user_agent.lower()

        features = {
            "has_user_agent": 1.0,
            "user_agent_length": len(user_agent),
        }

        # Browser detection
        browsers = ["chrome", "firefox", "safari", "edge", "opera"]
        browser_detected = any(browser in ua_lower for browser in browsers)
        features["browser_detected"] = float(browser_detected)

        # Mobile detection
        mobile_indicators = ["mobile", "android", "iphone", "ipad", "tablet"]
        features["is_mobile"] = float(
            any(indicator in ua_lower for indicator in mobile_indicators)
        )

        # Bot detection
        bot_indicators = ["bot", "crawler", "spider", "scraper", "curl", "wget"]
        features["is_bot"] = float(
            any(indicator in ua_lower for indicator in bot_indicators)
        )

        # Suspicious patterns
        features["has_suspicious_ua"] = float(
            len(user_agent) < 10
            or len(user_agent) > 500
            or user_agent.count("(") != user_agent.count(")")
            or "null" in ua_lower
            or "undefined" in ua_lower
        )

        # Risk scoring
        risk_factors = [
            features["is_bot"],
            features["has_suspicious_ua"],
            1.0 - features["browser_detected"],
        ]
        features["browser_risk"] = np.mean(risk_factors)
        features["os_risk"] = 0.5  # Placeholder for OS-based risk

        return features

    def _analyze_ip_address(self, ip_address: str) -> Dict[str, float]:
        """Analyze IP address characteristics."""
        features = {
            "ip_reputation": 0.5,  # Placeholder - would use IP reputation service
            "is_tor": 0.0,  # Placeholder - would check Tor exit nodes
            "is_vpn": 0.0,  # Placeholder - would check VPN databases
            "is_datacenter": 0.0,  # Placeholder - would check datacenter ranges
            "geolocation_risk": 0.5,  # Placeholder - would use geolocation service
        }

        # Basic IP analysis
        try:
            import ipaddress

            ip_obj = ipaddress.ip_address(ip_address)

            features["is_private"] = float(ip_obj.is_private)
            features["is_loopback"] = float(ip_obj.is_loopback)
            features["is_multicast"] = float(ip_obj.is_multicast)
            features["is_ipv6"] = float(ip_obj.version == 6)

        except ValueError:
            features.update(
                {
                    "is_private": 0.0,
                    "is_loopback": 0.0,
                    "is_multicast": 0.0,
                    "is_ipv6": 0.0,
                }
            )

        return features

    @staticmethod
    def _calculate_entropy(text: str) -> float:
        """Calculate Shannon entropy of text."""
        if not text:
            return 0.0

        # Count character frequencies
        char_counts = defaultdict(int)
        for char in text:
            char_counts[char] += 1

        # Calculate entropy
        length = len(text)
        entropy = 0.0
        for count in char_counts.values():
            probability = count / length
            entropy -= probability * np.log2(probability)

        return entropy


class FeatureEngineeringPipeline:
    """
    Main feature engineering pipeline that orchestrates all feature extractors.
    Provides real-time feature extraction with caching and parallel processing.
    """

    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        enable_caching: bool = True,
        cache_ttl: int = 300,
        max_workers: int = 4,
    ):
        self.redis_client = redis_client
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        self.max_workers = max_workers

        # Initialize feature extractors
        self.velocity_tracker = VelocityTracker(redis_client)
        self.behavioral_analyzer = BehavioralAnalyzer(redis_client)
        self.device_fingerprinter = DeviceFingerprinter(redis_client)

        # Feature cache
        self.feature_cache = {}

        # Performance metrics
        self.metrics = {
            "total_extractions": 0,
            "cache_hits": 0,
            "avg_extraction_time": 0.0,
        }

    def extract_features(
        self,
        transaction: Dict[str, Any],
        user_transactions: Optional[List[Dict]] = None,
        merchant_transactions: Optional[List[Dict]] = None,
        use_cache: bool = True,
    ) -> FeatureSet:
        """
        Extract comprehensive features for fraud detection.

        Args:
            transaction: Current transaction to analyze
            user_transactions: Historical user transactions
            merchant_transactions: Historical merchant transactions
            use_cache: Whether to use feature caching

        Returns:
            FeatureSet with all extracted features
        """
        start_time = datetime.now()

        # Generate cache key
        cache_key = self._generate_cache_key(transaction)

        # Check cache
        if use_cache and self.enable_caching and cache_key in self.feature_cache:
            cached_result = self.feature_cache[cache_key]
            if self._is_cache_valid(cached_result):
                self.metrics["cache_hits"] += 1
                return cached_result["features"]

        # Extract features in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit feature extraction tasks
            futures = {
                "transaction": executor.submit(
                    self._extract_transaction_features, transaction
                ),
                "velocity": executor.submit(
                    self._extract_velocity_features, transaction
                ),
                "behavioral": executor.submit(
                    self._extract_behavioral_features, transaction, user_transactions
                ),
                "device": executor.submit(self._extract_device_features, transaction),
                "temporal": executor.submit(
                    self._extract_temporal_features, transaction
                ),
                "network": executor.submit(self._extract_network_features, transaction),
            }

            # Collect results
            feature_results = {}
            for feature_type, future in futures.items():
                try:
                    feature_results[feature_type] = future.result(timeout=5.0)
                except Exception as e:
                    logger.error(f"Feature extraction failed for {feature_type}: {e}")
                    feature_results[feature_type] = {}

        # Create feature set
        feature_set = FeatureSet(
            transaction_features=feature_results.get("transaction", {}),
            velocity_features=feature_results.get("velocity", {}),
            behavioral_features=feature_results.get("behavioral", {}),
            device_features=feature_results.get("device", {}),
            temporal_features=feature_results.get("temporal", {}),
            network_features=feature_results.get("network", {}),
        )

        # Add derived features
        feature_set = self._add_derived_features(feature_set, transaction)

        # Cache result
        if use_cache and self.enable_caching:
            self.feature_cache[cache_key] = {
                "features": feature_set,
                "timestamp": datetime.now(),
            }

        # Update metrics
        extraction_time = (datetime.now() - start_time).total_seconds() * 1000
        self._update_metrics(extraction_time)

        return feature_set

    def _extract_transaction_features(
        self, transaction: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract basic transaction features."""
        amount = float(transaction.get("amount", 0))

        features = {
            "amount": amount,
            "amount_log": np.log1p(amount),
            "amount_sqrt": np.sqrt(amount),
            "amount_rounded": float(amount == round(amount)),
            "amount_cents": (amount * 100) % 100,
        }

        # Payment method encoding
        payment_methods = [
            "credit_card",
            "debit_card",
            "bank_transfer",
            "digital_wallet",
            "cash",
        ]
        current_method = transaction.get("payment_method", "")
        for method in payment_methods:
            features[f"payment_{method}"] = float(current_method == method)

        # Currency features
        currency = transaction.get("currency", "USD")
        features["is_usd"] = float(currency == "USD")
        features["is_foreign_currency"] = float(currency != "USD")

        return features

    def _extract_velocity_features(
        self, transaction: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract velocity-based features."""
        user_id = transaction.get("user_id", "")
        merchant_id = transaction.get("merchant_id", "")
        device_fp = transaction.get("device_fingerprint", "")
        ip_address = transaction.get("ip_address", "")

        features = {}

        # Update and get velocity features for each entity
        entities = [
            ("user", user_id),
            ("merchant", merchant_id),
            ("device", device_fp),
            ("ip", ip_address),
        ]

        for entity_type, entity_id in entities:
            if entity_id:
                # Update velocity tracking
                self.velocity_tracker.update_velocity(
                    entity_type, entity_id, transaction
                )

                # Get velocity features
                velocity_features = self.velocity_tracker.get_velocity_features(
                    entity_type, entity_id
                )

                # Add to main features with prefix
                for key, value in velocity_features.items():
                    features[f"{entity_type}_{key}"] = value

        return features

    def _extract_behavioral_features(
        self,
        transaction: Dict[str, Any],
        user_transactions: Optional[List[Dict]] = None,
    ) -> Dict[str, float]:
        """Extract behavioral analysis features."""
        user_id = transaction.get("user_id", "")

        # Update user profile
        self.behavioral_analyzer.update_user_profile(user_id, transaction)

        # Get behavioral features
        return self.behavioral_analyzer.get_behavioral_features(user_id, transaction)

    def _extract_device_features(self, transaction: Dict[str, Any]) -> Dict[str, float]:
        """Extract device and network features."""
        return self.device_fingerprinter.analyze_device(transaction)

    def _extract_temporal_features(
        self, transaction: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract temporal features."""
        try:
            timestamp = datetime.fromisoformat(
                transaction["timestamp"].replace("Z", "+00:00")
            )
        except:
            timestamp = datetime.now()

        features = {
            "hour": timestamp.hour,
            "day_of_week": timestamp.weekday(),
            "day_of_month": timestamp.day,
            "month": timestamp.month,
            "quarter": (timestamp.month - 1) // 3 + 1,
            "is_weekend": float(timestamp.weekday() >= 5),
            "is_business_hours": float(9 <= timestamp.hour <= 17),
            "is_night": float(timestamp.hour < 6 or timestamp.hour > 22),
            "is_early_morning": float(6 <= timestamp.hour < 9),
            "is_evening": float(18 <= timestamp.hour <= 22),
        }

        # Cyclical encoding for temporal features
        features.update(
            {
                "hour_sin": np.sin(2 * np.pi * timestamp.hour / 24),
                "hour_cos": np.cos(2 * np.pi * timestamp.hour / 24),
                "day_sin": np.sin(2 * np.pi * timestamp.weekday() / 7),
                "day_cos": np.cos(2 * np.pi * timestamp.weekday() / 7),
                "month_sin": np.sin(2 * np.pi * timestamp.month / 12),
                "month_cos": np.cos(2 * np.pi * timestamp.month / 12),
            }
        )

        return features

    def _extract_network_features(
        self, transaction: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract network-based features."""
        # Placeholder for network analysis features
        # In production, this would include:
        # - IP geolocation analysis
        # - Network topology features
        # - Connection pattern analysis

        return {
            "network_risk_score": 0.5,  # Placeholder
            "connection_anomaly": 0.0,  # Placeholder
            "routing_anomaly": 0.0,  # Placeholder
        }

    def _add_derived_features(
        self, feature_set: FeatureSet, transaction: Dict[str, Any]
    ) -> FeatureSet:
        """Add derived features based on combinations of base features."""
        # Get all features as dict
        all_features = feature_set.to_dict()

        # Amount-based derived features
        user_avg_amount = all_features.get("velocity_features_user_amount_avg_24h", 1.0)
        merchant_avg_amount = all_features.get(
            "velocity_features_merchant_amount_avg_24h", 1.0
        )
        current_amount = all_features.get("transaction_features_amount", 0.0)

        derived_features = {
            "amount_vs_user_avg": current_amount / max(user_avg_amount, 1.0),
            "amount_vs_merchant_avg": current_amount / max(merchant_avg_amount, 1.0),
            "is_large_amount": float(current_amount > user_avg_amount * 3),
            "is_small_amount": float(current_amount < user_avg_amount * 0.1),
        }

        # Velocity-based derived features
        user_count_1h = all_features.get("velocity_features_user_count_1h", 0.0)
        user_count_24h = all_features.get("velocity_features_user_count_24h", 0.0)

        derived_features.update(
            {
                "velocity_acceleration": user_count_1h / max(user_count_24h / 24, 1.0),
                "is_high_velocity": float(user_count_1h > 10),
                "is_burst_activity": float(user_count_1h > user_count_24h * 0.5),
            }
        )

        # Behavioral derived features
        overall_familiarity = all_features.get(
            "behavioral_features_overall_familiarity", 0.5
        )
        anomaly_score = all_features.get("behavioral_features_anomaly_score", 0.5)

        derived_features.update(
            {
                "behavioral_risk": (1.0 - overall_familiarity) * anomaly_score,
                "is_new_pattern": float(overall_familiarity < 0.1),
                "is_familiar_pattern": float(overall_familiarity > 0.8),
            }
        )

        # Add derived features to feature set
        for key, value in derived_features.items():
            feature_set.transaction_features[f"derived_{key}"] = value

        return feature_set

    def _generate_cache_key(self, transaction: Dict[str, Any]) -> str:
        """Generate cache key for transaction."""
        key_data = {
            "transaction_id": transaction.get("transaction_id"),
            "user_id": transaction.get("user_id"),
            "amount": transaction.get("amount"),
            "timestamp": transaction.get("timestamp"),
            "merchant_id": transaction.get("merchant_id"),
        }
        return hashlib.md5(str(sorted(key_data.items())).encode()).hexdigest()

    def _is_cache_valid(self, cached_result: Dict) -> bool:
        """Check if cached result is still valid."""
        age = (datetime.now() - cached_result["timestamp"]).total_seconds()
        return age < self.cache_ttl

    def _update_metrics(self, extraction_time: float):
        """Update performance metrics."""
        self.metrics["total_extractions"] += 1

        # Update average extraction time (exponential moving average)
        alpha = 0.1
        self.metrics["avg_extraction_time"] = (
            alpha * extraction_time + (1 - alpha) * self.metrics["avg_extraction_time"]
        )

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        total_extractions = self.metrics["total_extractions"]
        cache_hits = self.metrics["cache_hits"]

        return {
            "total_extractions": total_extractions,
            "cache_hit_rate": cache_hits / max(total_extractions, 1),
            "avg_extraction_time_ms": self.metrics["avg_extraction_time"],
            "extractions_per_second": 1000
            / max(self.metrics["avg_extraction_time"], 1),
        }

    def clear_cache(self):
        """Clear feature cache."""
        self.feature_cache.clear()
        logger.info("Feature cache cleared")

    def get_feature_names(self) -> List[str]:
        """Get list of all possible feature names."""
        # This would return a comprehensive list of all feature names
        # Used for model training and inference
        feature_names = []

        # Add feature names from each category
        categories = [
            "transaction_features",
            "velocity_features",
            "behavioral_features",
            "device_features",
            "temporal_features",
            "network_features",
        ]

        # This is a simplified version - in production, you'd maintain
        # a comprehensive list of all possible feature names
        for category in categories:
            feature_names.extend(
                [
                    f"{category}_amount",
                    f"{category}_count_1h",
                    f"{category}_risk_score",
                    # ... add all possible feature names
                ]
            )

        return sorted(feature_names)
