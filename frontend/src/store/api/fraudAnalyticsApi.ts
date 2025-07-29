/**
 * RTK Query API for fraud analytics platform
 * Handles all API communication with type safety and caching
 */

import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';
import type { RootState } from '../index';

// Types
export interface Transaction {
  id: string;
  transaction_id: string;
  user_id: string;
  amount: number;
  currency: string;
  merchant_id: string;
  payment_method: string;
  status: 'pending' | 'approved' | 'declined' | 'flagged';
  fraud_score?: number;
  risk_level?: string;
  ip_address: string;
  user_agent?: string;
  device_fingerprint?: string;
  timestamp: string;
  created_at: string;
  updated_at: string;
}

export interface FraudAnalysisResult {
  fraud_probability: number;
  risk_score: number;
  confidence: number;
  risk_level: string;
  recommendation: string;
  ml_results: {
    ml_fraud_probability: number;
    ml_risk_score: number;
    ml_confidence: number;
    model_predictions: Record<string, any>;
    ensemble_weights: Record<string, number>;
    models_used: string[];
  };
  rule_results: Array<{
    rule_id: string;
    rule_name: string;
    triggered: boolean;
    confidence: number;
    action?: string;
  }>;
  performance_metrics: {
    total_time_ms: number;
    feature_extraction_time_ms: number;
    ml_prediction_time_ms: number;
    rules_evaluation_time_ms: number;
  };
  explanation: {
    summary: string;
    key_factors: string[];
    model_contributions: Record<string, any>;
    rule_contributions: Array<any>;
  };
}

export interface Case {
  id: string;
  case_number: string;
  title: string;
  description: string;
  status: 'open' | 'investigating' | 'resolved' | 'closed';
  priority: 'low' | 'medium' | 'high' | 'critical';
  transaction_id: string;
  assigned_to?: string;
  created_at: string;
  updated_at: string;
  resolved_at?: string;
}

export interface FraudRule {
  id: string;
  name: string;
  description: string;
  conditions: Record<string, any>;
  action: 'flag' | 'decline' | 'alert';
  is_active: boolean;
  priority: number;
  created_at: string;
  updated_at: string;
}

export interface DashboardStats {
  total_transactions: number;
  flagged_transactions: number;
  open_cases: number;
  average_fraud_score: number;
  fraud_rate: number;
  daily_stats: Array<{
    date: string;
    transactions: number;
    fraud_detected: number;
    fraud_rate: number;
  }>;
  model_performance: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
  };
  top_risk_factors: Array<{
    factor: string;
    impact: number;
    frequency: number;
  }>;
}

export interface User {
  id: string;
  username: string;
  email: string;
  company_name: string;
  is_verified: boolean;
  created_at: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface LoginResponse {
  access: string;
  refresh: string;
  user: User;
}

// API slice
export const fraudAnalyticsApi = createApi({
  reducerPath: 'fraudAnalyticsApi',
  baseQuery: fetchBaseQuery({
    baseUrl: '/api/v1/',
    prepareHeaders: (headers, { getState }) => {
      const token = (getState() as RootState).auth.token;
      if (token) {
        headers.set('authorization', `Bearer ${token}`);
      }
      headers.set('content-type', 'application/json');
      return headers;
    },
  }),
  tagTypes: ['Transaction', 'Case', 'Rule', 'User', 'Dashboard'],
  endpoints: (builder) => ({
    // Authentication
    login: builder.mutation<LoginResponse, LoginRequest>({
      query: (credentials) => ({
        url: 'auth/token/',
        method: 'POST',
        body: credentials,
      }),
      invalidatesTags: ['User'],
    }),

    register: builder.mutation<User, Partial<User> & { password: string }>({
      query: (userData) => ({
        url: 'auth/register/',
        method: 'POST',
        body: userData,
      }),
    }),

    getProfile: builder.query<User, void>({
      query: () => 'auth/profile/',
      providesTags: ['User'],
    }),

    // Dashboard
    getDashboardStats: builder.query<DashboardStats, void>({
      query: () => 'analytics/dashboard/',
      providesTags: ['Dashboard'],
    }),

    // Transactions
    getTransactions: builder.query<
      { results: Transaction[]; count: number; next?: string; previous?: string },
      { page?: number; page_size?: number; status?: string; search?: string }
    >({
      query: (params) => ({
        url: 'transactions/',
        params,
      }),
      providesTags: (result) =>
        result
          ? [
              ...result.results.map(({ id }) => ({ type: 'Transaction' as const, id })),
              { type: 'Transaction', id: 'LIST' },
            ]
          : [{ type: 'Transaction', id: 'LIST' }],
    }),

    getTransaction: builder.query<Transaction, string>({
      query: (id) => `transactions/${id}/`,
      providesTags: (result, error, id) => [{ type: 'Transaction', id }],
    }),

    analyzeTransaction: builder.mutation<FraudAnalysisResult, Partial<Transaction>>({
      query: (transactionData) => ({
        url: 'fraud/analyze/',
        method: 'POST',
        body: transactionData,
      }),
      invalidatesTags: ['Transaction', 'Dashboard'],
    }),

    updateTransactionStatus: builder.mutation<
      Transaction,
      { id: string; status: Transaction['status'] }
    >({
      query: ({ id, status }) => ({
        url: `transactions/${id}/`,
        method: 'PATCH',
        body: { status },
      }),
      invalidatesTags: (result, error, { id }) => [
        { type: 'Transaction', id },
        { type: 'Transaction', id: 'LIST' },
        'Dashboard',
      ],
    }),

    // Cases
    getCases: builder.query<
      { results: Case[]; count: number; next?: string; previous?: string },
      { page?: number; page_size?: number; status?: string; priority?: string }
    >({
      query: (params) => ({
        url: 'cases/',
        params,
      }),
      providesTags: (result) =>
        result
          ? [
              ...result.results.map(({ id }) => ({ type: 'Case' as const, id })),
              { type: 'Case', id: 'LIST' },
            ]
          : [{ type: 'Case', id: 'LIST' }],
    }),

    getCase: builder.query<Case, string>({
      query: (id) => `cases/${id}/`,
      providesTags: (result, error, id) => [{ type: 'Case', id }],
    }),

    createCase: builder.mutation<Case, Partial<Case>>({
      query: (caseData) => ({
        url: 'cases/',
        method: 'POST',
        body: caseData,
      }),
      invalidatesTags: [{ type: 'Case', id: 'LIST' }, 'Dashboard'],
    }),

    updateCase: builder.mutation<Case, { id: string; data: Partial<Case> }>({
      query: ({ id, data }) => ({
        url: `cases/${id}/`,
        method: 'PATCH',
        body: data,
      }),
      invalidatesTags: (result, error, { id }) => [
        { type: 'Case', id },
        { type: 'Case', id: 'LIST' },
        'Dashboard',
      ],
    }),

    // Fraud Rules
    getFraudRules: builder.query<
      { results: FraudRule[]; count: number },
      { page?: number; page_size?: number; is_active?: boolean }
    >({
      query: (params) => ({
        url: 'fraud/rules/',
        params,
      }),
      providesTags: (result) =>
        result
          ? [
              ...result.results.map(({ id }) => ({ type: 'Rule' as const, id })),
              { type: 'Rule', id: 'LIST' },
            ]
          : [{ type: 'Rule', id: 'LIST' }],
    }),

    createFraudRule: builder.mutation<FraudRule, Partial<FraudRule>>({
      query: (ruleData) => ({
        url: 'fraud/rules/',
        method: 'POST',
        body: ruleData,
      }),
      invalidatesTags: [{ type: 'Rule', id: 'LIST' }],
    }),

    updateFraudRule: builder.mutation<FraudRule, { id: string; data: Partial<FraudRule> }>({
      query: ({ id, data }) => ({
        url: `fraud/rules/${id}/`,
        method: 'PATCH',
        body: data,
      }),
      invalidatesTags: (result, error, { id }) => [
        { type: 'Rule', id },
        { type: 'Rule', id: 'LIST' },
      ],
    }),

    deleteFraudRule: builder.mutation<void, string>({
      query: (id) => ({
        url: `fraud/rules/${id}/`,
        method: 'DELETE',
      }),
      invalidatesTags: (result, error, id) => [
        { type: 'Rule', id },
        { type: 'Rule', id: 'LIST' },
      ],
    }),

    // Fraud Alerts
    getFraudAlerts: builder.query<
      { results: any[]; count: number },
      { page?: number; page_size?: number; is_resolved?: boolean }
    >({
      query: (params) => ({
        url: 'fraud/alerts/',
        params,
      }),
    }),

    // Webhooks
    getWebhooks: builder.query<{ results: any[]; count: number }, void>({
      query: () => 'webhooks/',
    }),

    createWebhook: builder.mutation<any, { name: string; url: string; events: string[] }>({
      query: (webhookData) => ({
        url: 'webhooks/',
        method: 'POST',
        body: webhookData,
      }),
    }),
  }),
});

// Export hooks
export const {
  // Auth
  useLoginMutation,
  useRegisterMutation,
  useGetProfileQuery,

  // Dashboard
  useGetDashboardStatsQuery,

  // Transactions
  useGetTransactionsQuery,
  useGetTransactionQuery,
  useAnalyzeTransactionMutation,
  useUpdateTransactionStatusMutation,

  // Cases
  useGetCasesQuery,
  useGetCaseQuery,
  useCreateCaseMutation,
  useUpdateCaseMutation,

  // Rules
  useGetFraudRulesQuery,
  useCreateFraudRuleMutation,
  useUpdateFraudRuleMutation,
  useDeleteFraudRuleMutation,

  // Alerts
  useGetFraudAlertsQuery,

  // Webhooks
  useGetWebhooksQuery,
  useCreateWebhookMutation,
} = fraudAnalyticsApi;