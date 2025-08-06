# 📚 Fraud Analytics Platform - API Reference

> Complete API documentation for enterprise-grade fraud detection and transaction monitoring

## 🔗 Base URL

```
Production: https://api.fraudanalytics.com/v1
Staging: https://staging-api.fraudanalytics.com/v1
Development: http://localhost:8000/api/v1
```

## 🔐 Authentication

The API uses JWT (JSON Web Token) authentication. Include the token in the Authorization header:

```http
Authorization: Bearer <your_jwt_token>
```

### Get JWT Token

```http
POST /auth/token/
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "your_password"
}
```

**Response:**
```json
{
  "access": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

## 📊 Rate Limits

| Tier | Requests/Minute | Requests/Hour | Requests/Day |
|------|----------------|---------------|--------------|
| Free | 100 | 1,000 | 10,000 |
| Pro | 1,000 | 10,000 | 100,000 |
| Enterprise | Custom | Custom | Custom |

## 🚀 Core Endpoints

### 1. Fraud Analysis

Analyze transactions for fraud indicators in real-time.

```http
POST /fraud/analyze/
Authorization: Bearer <token>
Content-Type: application/json

{
  "transaction_id": "txn_123456789",
  "user_id": "user_987654321",
  "amount": 150.00,
  "currency": "USD",
  "merchant_id": "merchant_amazon",
  "payment_method": "credit_card",
  "ip_address": "192.168.1.100",
  "user_agent": "Mozilla/5.0...",
  "device_fingerprint": "fp_abc123",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Response:**
```json
{
  "fraud_score": 25.5,
  "risk_level": "low",
  "recommendation": "approve",
  "confidence": 0.87,
  "rule_results": [
    {
      "rule_name": "Velocity Check",
      "triggered": false,
      "score_impact": 0
    }
  ],
  "ml_predictions": {
    "xgboost_score": 0.23,
    "lstm_score": 0.28,
    "gnn_score": 0.25,
    "transformer_score": 0.26,
    "ensemble_score": 0.255
  },
  "feature_analysis": {
    "amount_percentile": 45.2,
    "velocity_score": 12.1,
    "geographic_risk": 5.0,
    "device_risk": 8.3
  },
  "processing_time_ms": 45,
  "analysis_timestamp": "2024-01-15T10:30:01.234Z"
}
```

### 2. Transaction Management

#### Create Transaction

```http
POST /transactions/
Authorization: Bearer <token>
Content-Type: application/json

{
  "transaction_id": "txn_unique_123",
  "user_id": "user_456",
  "amount": "99.99",
  "currency": "USD",
  "merchant_id": "merchant_store",
  "payment_method": "credit_card",
  "ip_address": "203.0.113.1",
  "timestamp": "2024-01-15T14:30:00Z"
}
```

#### List Transactions

```http
GET /transactions/?page=1&page_size=20&status=flagged
Authorization: Bearer <token>
```

**Response:**
```json
{
  "count": 150,
  "next": "http://api.example.com/v1/transactions/?page=2",
  "previous": null,
  "results": [
    {
      "id": "uuid-123",
      "transaction_id": "txn_123",
      "amount": "99.99",
      "currency": "USD",
      "status": "flagged",
      "fraud_score": 75.2,
      "risk_level": "high",
      "created_at": "2024-01-15T14:30:00Z"
    }
  ]
}
```

#### Get Transaction Details

```http
GET /transactions/{transaction_id}/
Authorization: Bearer <token>
```

### 3. Fraud Rules Management

#### Create Fraud Rule

```http
POST /fraud/rules/
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "High Amount Alert",
  "description": "Flag transactions over $1000",
  "conditions": {
    "amount_threshold": 1000,
    "currency": "USD",
    "time_window": "1h"
  },
  "action": "flag",
  "priority": 1,
  "is_active": true
}
```

#### List Fraud Rules

```http
GET /fraud/rules/?is_active=true
Authorization: Bearer <token>
```

### 4. Case Management

#### Create Investigation Case

```http
POST /cases/
Authorization: Bearer <token>
Content-Type: application/json

{
  "title": "Suspicious High-Value Transaction",
  "description": "Multiple high-value transactions from new user",
  "transaction_id": "txn_suspicious_123",
  "priority": "high",
  "status": "open"
}
```

#### Update Case Status

```http
PATCH /cases/{case_id}/
Authorization: Bearer <token>
Content-Type: application/json

{
  "status": "investigating",
  "assigned_to": "analyst_user_id",
  "notes": "Started investigation, checking user history"
}
```

### 5. Analytics & Reporting

#### Dashboard Statistics

```http
GET /analytics/dashboard/
Authorization: Bearer <token>
```

**Response:**
```json
{
  "total_transactions": 15420,
  "flagged_transactions": 234,
  "open_cases": 12,
  "resolved_cases": 89,
  "average_fraud_score": 23.5,
  "fraud_rate": 1.52,
  "daily_volume": 1250,
  "weekly_trend": 5.2,
  "top_risk_factors": [
    {"factor": "high_amount", "count": 45},
    {"factor": "new_device", "count": 32},
    {"factor": "unusual_location", "count": 28}
  ],
  "performance_metrics": {
    "avg_processing_time_ms": 42,
    "accuracy_rate": 94.7,
    "false_positive_rate": 2.1
  }
}
```

#### Custom Reports

```http
POST /analytics/reports/
Authorization: Bearer <token>
Content-Type: application/json

{
  "report_type": "fraud_summary",
  "date_range": {
    "start": "2024-01-01",
    "end": "2024-01-31"
  },
  "filters": {
    "risk_level": ["medium", "high"],
    "merchant_ids": ["merchant_1", "merchant_2"]
  },
  "format": "json"
}
```

### 6. Webhook Management

#### Register Webhook

```http
POST /webhooks/
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "Fraud Alert Webhook",
  "url": "https://your-app.com/webhooks/fraud-alerts",
  "events": [
    "transaction.flagged",
    "case.created",
    "case.resolved"
  ],
  "secret": "your_webhook_secret_key",
  "is_active": true
}
```

#### Webhook Events

| Event | Description | Payload |
|-------|-------------|---------|
| `transaction.flagged` | Transaction flagged as fraudulent | Transaction details + fraud analysis |
| `transaction.approved` | Transaction approved after review | Transaction details |
| `case.created` | New fraud case created | Case details |
| `case.updated` | Case status updated | Case details + changes |
| `case.resolved` | Case marked as resolved | Case details + resolution |
| `rule.triggered` | Fraud rule triggered | Rule details + transaction |

## 🔍 Advanced Features

### Batch Processing

Process multiple transactions in a single request:

```http
POST /fraud/analyze/batch/
Authorization: Bearer <token>
Content-Type: application/json

{
  "transactions": [
    {
      "transaction_id": "txn_1",
      "amount": 100.00,
      // ... other fields
    },
    {
      "transaction_id": "txn_2",
      "amount": 250.00,
      // ... other fields
    }
  ]
}
```

### Real-time Streaming

Connect to real-time fraud alerts via WebSocket:

```javascript
const ws = new WebSocket('wss://api.fraudanalytics.com/v1/stream/');
ws.onmessage = function(event) {
  const alert = JSON.parse(event.data);
  console.log('Fraud alert:', alert);
};
```

### Machine Learning Model Management

#### Get Model Performance

```http
GET /ml/models/performance/
Authorization: Bearer <token>
```

#### Update Model Configuration

```http
POST /ml/models/config/
Authorization: Bearer <token>
Content-Type: application/json

{
  "model_type": "ensemble",
  "parameters": {
    "fraud_threshold": 0.7,
    "confidence_threshold": 0.8,
    "feature_weights": {
      "amount": 0.3,
      "velocity": 0.25,
      "geographic": 0.2,
      "device": 0.25
    }
  }
}
```

## 📋 Response Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request - Invalid input |
| 401 | Unauthorized - Invalid or missing token |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

## 🛡️ Security

### API Key Authentication (Alternative)

For server-to-server communication, you can use API keys:

```http
Authorization: ApiKey your_api_key_here
```

### Request Signing

For enhanced security, sign requests with HMAC-SHA256:

```http
X-Signature: sha256=<hmac_signature>
X-Timestamp: 1642234567
```

### IP Whitelisting

Configure IP whitelisting in your account settings for additional security.

## 📊 SDKs and Libraries

### Python SDK

```python
from fraud_analytics import FraudAnalyticsClient

client = FraudAnalyticsClient(
    api_key='your_api_key',
    base_url='https://api.fraudanalytics.com/v1'
)

result = client.analyze_transaction({
    'transaction_id': 'txn_123',
    'amount': 100.00,
    'user_id': 'user_456'
})

print(f"Fraud score: {result.fraud_score}")
```

### JavaScript SDK

```javascript
import { FraudAnalytics } from '@fraud-analytics/js-sdk';

const client = new FraudAnalytics({
  apiKey: 'your_api_key',
  baseUrl: 'https://api.fraudanalytics.com/v1'
});

const result = await client.analyzeTransaction({
  transactionId: 'txn_123',
  amount: 100.00,
  userId: 'user_456'
});

console.log(`Fraud score: ${result.fraudScore}`);
```

## 🔧 Testing

### Sandbox Environment

Use the sandbox environment for testing:

```
Sandbox URL: https://sandbox-api.fraudanalytics.com/v1
```

### Test Data

Use these test transaction IDs for different scenarios:

- `test_approved_txn` - Always returns low fraud score
- `test_flagged_txn` - Always returns high fraud score
- `test_timeout_txn` - Simulates processing timeout
- `test_error_txn` - Simulates processing error

## 📞 Support

- **Documentation**: https://docs.fraudanalytics.com
- **API Status**: https://status.fraudanalytics.com
- **Support Email**: api-support@fraudanalytics.com
- **Developer Forum**: https://community.fraudanalytics.com

## 📝 Changelog

### v1.2.0 (Latest)
- Added batch processing endpoint
- Enhanced ML model performance
- Improved webhook reliability
- Added real-time streaming support

### v1.1.0
- Added case management endpoints
- Enhanced fraud rule engine
- Improved analytics dashboard
- Added webhook support

### v1.0.0
- Initial API release
- Core fraud analysis functionality
- Transaction management
- Basic reporting features