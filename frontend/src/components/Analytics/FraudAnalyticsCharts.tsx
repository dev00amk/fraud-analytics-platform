/**
 * Advanced fraud analytics charts with interactive visualizations
 * Uses Recharts and D3 for comprehensive fraud pattern analysis
 */

import React, { useMemo, useState } from 'react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar
} from 'recharts';
import { motion } from 'framer-motion';
import { 
  ArrowTrendingUpIcon, 
  ArrowTrendingDownIcon,
  ExclamationTriangleIcon,
  ChartBarIcon
} from '@heroicons/react/24/outline';
import { DashboardStats } from '../../store/api/fraudAnalyticsApi';

interface FraudAnalyticsChartsProps {
  data: DashboardStats;
  timeRange: '24h' | '7d' | '30d' | '90d';
  onTimeRangeChange: (range: '24h' | '7d' | '30d' | '90d') => void;
}

const COLORS = {
  primary: '#3B82F6',
  secondary: '#10B981',
  danger: '#EF4444',
  warning: '#F59E0B',
  info: '#06B6D4',
  purple: '#8B5CF6',
  pink: '#EC4899',
  gray: '#6B7280'
};

const CHART_COLORS = [
  COLORS.primary,
  COLORS.secondary,
  COLORS.danger,
  COLORS.warning,
  COLORS.info,
  COLORS.purple,
  COLORS.pink
];

const FraudAnalyticsCharts: React.FC<FraudAnalyticsChartsProps> = ({
  data,
  timeRange,
  onTimeRangeChange
}) => {
  const [selectedChart, setSelectedChart] = useState<string | null>(null);

  // Process data for different chart types
  const processedData = useMemo(() => {
    const dailyData = data.daily_stats.map(stat => ({
      ...stat,
      date: new Date(stat.date).toLocaleDateString('en-US', { 
        month: 'short', 
        day: 'numeric' 
      }),
      fraud_rate_percentage: stat.fraud_rate * 100,
      legitimate_transactions: stat.transactions - stat.fraud_detected
    }));

    const riskDistribution = [
      { name: 'Low Risk', value: 65, color: COLORS.secondary },
      { name: 'Medium Risk', value: 25, color: COLORS.warning },
      { name: 'High Risk', value: 8, color: COLORS.danger },
      { name: 'Critical Risk', value: 2, color: '#DC2626' }
    ];

    const modelPerformanceData = [
      { model: 'XGBoost', accuracy: 0.92, precision: 0.89, recall: 0.94, f1: 0.91 },
      { model: 'LSTM', accuracy: 0.88, precision: 0.85, recall: 0.91, f1: 0.88 },
      { model: 'GNN', accuracy: 0.94, precision: 0.92, recall: 0.96, f1: 0.94 },
      { model: 'Transformer', accuracy: 0.96, precision: 0.94, recall: 0.98, f1: 0.96 },
      { model: 'Ensemble', accuracy: 0.97, precision: 0.95, recall: 0.99, f1: 0.97 }
    ];

    const fraudPatterns = [
      { pattern: 'Velocity Abuse', frequency: 45, impact: 8.2 },
      { pattern: 'Account Takeover', frequency: 32, impact: 9.1 },
      { pattern: 'Card Testing', frequency: 28, impact: 3.4 },
      { pattern: 'Synthetic Identity', frequency: 18, impact: 7.8 },
      { pattern: 'Friendly Fraud', frequency: 15, impact: 4.2 },
      { pattern: 'BIN Attack', frequency: 12, impact: 6.5 }
    ];

    return {
      dailyData,
      riskDistribution,
      modelPerformanceData,
      fraudPatterns
    };
  }, [data]);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
          <p className="font-medium text-gray-900">{label}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} style={{ color: entry.color }} className="text-sm">
              {entry.name}: {typeof entry.value === 'number' ? entry.value.toLocaleString() : entry.value}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  const MetricCard = ({ 
    title, 
    value, 
    change, 
    icon: Icon, 
    color = 'blue' 
  }: {
    title: string;
    value: string | number;
    change?: number;
    icon: React.ComponentType<any>;
    color?: string;
  }) => (
    <motion.div
      whileHover={{ scale: 1.02 }}
      className="bg-white p-6 rounded-lg border border-gray-200 shadow-sm"
    >
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-bold text-gray-900">{value}</p>
          {change !== undefined && (
            <div className="flex items-center mt-1">
              {change >= 0 ? (
                <ArrowTrendingUpIcon className="h-4 w-4 text-green-500 mr-1" />
              ) : (
                <ArrowTrendingDownIcon className="h-4 w-4 text-red-500 mr-1" />
              )}
              <span className={`text-sm ${change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {Math.abs(change)}%
              </span>
            </div>
          )}
        </div>
        <Icon className={`h-8 w-8 text-${color}-500`} />
      </div>
    </motion.div>
  );

  return (
    <div className="space-y-6">
      {/* Time Range Selector */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900">Fraud Analytics</h2>
        <div className="flex space-x-1 bg-gray-100 rounded-lg p-1">
          {(['24h', '7d', '30d', '90d'] as const).map((range) => (
            <button
              key={range}
              onClick={() => onTimeRangeChange(range)}
              className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
                timeRange === range
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              {range}
            </button>
          ))}
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Total Transactions"
          value={data.total_transactions.toLocaleString()}
          change={12.5}
          icon={ChartBarIcon}
          color="blue"
        />
        <MetricCard
          title="Fraud Rate"
          value={`${(data.fraud_rate * 100).toFixed(2)}%`}
          change={-2.3}
          icon={ExclamationTriangleIcon}
          color="red"
        />
        <MetricCard
          title="Model Accuracy"
          value={`${(data.model_performance.accuracy * 100).toFixed(1)}%`}
          change={1.8}
          icon={ChartBarIcon}
          color="green"
        />
        <MetricCard
          title="Open Cases"
          value={data.open_cases}
          change={-5.2}
          icon={ExclamationTriangleIcon}
          color="yellow"
        />
      </div>

      {/* Main Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Fraud Trend Chart */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white p-6 rounded-lg border border-gray-200 shadow-sm"
        >
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Fraud Detection Trends
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={processedData.dailyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="date" stroke="#6b7280" />
              <YAxis stroke="#6b7280" />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              <Area
                type="monotone"
                dataKey="transactions"
                stackId="1"
                stroke={COLORS.primary}
                fill={COLORS.primary}
                fillOpacity={0.6}
                name="Total Transactions"
              />
              <Area
                type="monotone"
                dataKey="fraud_detected"
                stackId="2"
                stroke={COLORS.danger}
                fill={COLORS.danger}
                fillOpacity={0.8}
                name="Fraud Detected"
              />
            </AreaChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Risk Distribution */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-white p-6 rounded-lg border border-gray-200 shadow-sm"
        >
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Risk Level Distribution
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={processedData.riskDistribution}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={120}
                paddingAngle={5}
                dataKey="value"
              >
                {processedData.riskDistribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip 
                formatter={(value: number) => [`${value}%`, 'Percentage']}
                content={<CustomTooltip />}
              />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Model Performance Comparison */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-white p-6 rounded-lg border border-gray-200 shadow-sm"
        >
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            ML Model Performance
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={processedData.modelPerformanceData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="model" />
              <PolarRadiusAxis 
                angle={90} 
                domain={[0.8, 1]} 
                tick={false}
              />
              <Radar
                name="Accuracy"
                dataKey="accuracy"
                stroke={COLORS.primary}
                fill={COLORS.primary}
                fillOpacity={0.3}
              />
              <Radar
                name="Precision"
                dataKey="precision"
                stroke={COLORS.secondary}
                fill={COLORS.secondary}
                fillOpacity={0.3}
              />
              <Radar
                name="Recall"
                dataKey="recall"
                stroke={COLORS.danger}
                fill={COLORS.danger}
                fillOpacity={0.3}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
            </RadarChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Fraud Patterns Analysis */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-white p-6 rounded-lg border border-gray-200 shadow-sm"
        >
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Fraud Pattern Analysis
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart data={processedData.fraudPatterns}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis 
                dataKey="frequency" 
                name="Frequency"
                stroke="#6b7280"
                label={{ value: 'Frequency', position: 'insideBottom', offset: -5 }}
              />
              <YAxis 
                dataKey="impact" 
                name="Impact Score"
                stroke="#6b7280"
                label={{ value: 'Impact Score', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip 
                cursor={{ strokeDasharray: '3 3' }}
                formatter={(value: number, name: string) => [
                  value,
                  name === 'frequency' ? 'Frequency' : 'Impact Score'
                ]}
                labelFormatter={(label: string) => `Pattern: ${label}`}
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
                        <p className="font-medium text-gray-900">{data.pattern}</p>
                        <p className="text-sm text-blue-600">Frequency: {data.frequency}</p>
                        <p className="text-sm text-red-600">Impact: {data.impact}</p>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Scatter 
                dataKey="impact" 
                fill={COLORS.primary}
                stroke={COLORS.primary}
                strokeWidth={2}
              />
            </ScatterChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      {/* Detailed Fraud Rate Timeline */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="bg-white p-6 rounded-lg border border-gray-200 shadow-sm"
      >
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Detailed Fraud Rate Timeline
        </h3>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={processedData.dailyData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="date" stroke="#6b7280" />
            <YAxis 
              yAxisId="left"
              stroke="#6b7280"
              label={{ value: 'Transactions', angle: -90, position: 'insideLeft' }}
            />
            <YAxis 
              yAxisId="right" 
              orientation="right"
              stroke="#6b7280"
              label={{ value: 'Fraud Rate (%)', angle: 90, position: 'insideRight' }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Bar
              yAxisId="left"
              dataKey="legitimate_transactions"
              fill={COLORS.secondary}
              fillOpacity={0.6}
              name="Legitimate Transactions"
            />
            <Bar
              yAxisId="left"
              dataKey="fraud_detected"
              fill={COLORS.danger}
              fillOpacity={0.8}
              name="Fraud Detected"
            />
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="fraud_rate_percentage"
              stroke={COLORS.warning}
              strokeWidth={3}
              dot={{ fill: COLORS.warning, strokeWidth: 2, r: 4 }}
              name="Fraud Rate %"
            />
          </LineChart>
        </ResponsiveContainer>
      </motion.div>

      {/* Top Risk Factors */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="bg-white p-6 rounded-lg border border-gray-200 shadow-sm"
      >
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Top Risk Factors
        </h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart 
            data={data.top_risk_factors}
            layout="horizontal"
            margin={{ left: 100 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis type="number" stroke="#6b7280" />
            <YAxis 
              type="category" 
              dataKey="factor" 
              stroke="#6b7280"
              width={100}
            />
            <Tooltip content={<CustomTooltip />} />
            <Bar 
              dataKey="impact" 
              fill={COLORS.primary}
              name="Impact Score"
              radius={[0, 4, 4, 0]}
            />
          </BarChart>
        </ResponsiveContainer>
      </motion.div>
    </div>
  );
};

export default FraudAnalyticsCharts;