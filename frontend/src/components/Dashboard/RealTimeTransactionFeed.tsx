/**
 * Real-time transaction feed with WebSocket integration
 * Shows live transactions with fraud alerts and risk indicators
 */

import React, { useEffect, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ExclamationTriangleIcon, 
  CheckCircleIcon, 
  ClockIcon,
  XCircleIcon,
  EyeIcon,
  ArrowPathIcon
} from '@heroicons/react/24/outline';
import { format } from 'date-fns';
import { io, Socket } from 'socket.io-client';
import { Transaction, FraudAnalysisResult } from '../../store/api/fraudAnalyticsApi';
import { useAppSelector } from '../../store/hooks';
import toast from 'react-hot-toast';

interface TransactionWithAnalysis extends Transaction {
  analysis?: FraudAnalysisResult;
  isNew?: boolean;
}

interface RealTimeTransactionFeedProps {
  maxItems?: number;
  showFilters?: boolean;
  onTransactionClick?: (transaction: TransactionWithAnalysis) => void;
}

const RealTimeTransactionFeed: React.FC<RealTimeTransactionFeedProps> = ({
  maxItems = 50,
  showFilters = true,
  onTransactionClick
}) => {
  const [transactions, setTransactions] = useState<TransactionWithAnalysis[]>([]);
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [filter, setFilter] = useState<'all' | 'flagged' | 'high_risk'>('all');
  const [isPaused, setIsPaused] = useState(false);
  
  const { token } = useAppSelector((state) => state.auth);

  // WebSocket connection
  useEffect(() => {
    if (!token) return;

    const newSocket = io('ws://localhost:8000', {
      auth: { token },
      transports: ['websocket']
    });

    newSocket.on('connect', () => {
      setIsConnected(true);
      toast.success('Connected to real-time feed');
    });

    newSocket.on('disconnect', () => {
      setIsConnected(false);
      toast.error('Disconnected from real-time feed');
    });

    newSocket.on('transaction_analyzed', (data: TransactionWithAnalysis) => {
      if (!isPaused) {
        setTransactions(prev => {
          const newTransaction = { ...data, isNew: true };
          const updated = [newTransaction, ...prev.slice(0, maxItems - 1)];
          
          // Remove new flag after animation
          setTimeout(() => {
            setTransactions(current => 
              current.map(t => 
                t.id === newTransaction.id ? { ...t, isNew: false } : t
              )
            );
          }, 2000);
          
          return updated;
        });

        // Show toast for high-risk transactions
        if (data.analysis && data.analysis.risk_level === 'high') {
          toast.error(`High-risk transaction detected: ${data.transaction_id}`);
        }
      }
    });

    newSocket.on('fraud_alert', (alert: any) => {
      toast.error(`Fraud Alert: ${alert.message}`);
    });

    setSocket(newSocket);

    return () => {
      newSocket.close();
    };
  }, [token, maxItems, isPaused]);

  const getRiskColor = (riskLevel?: string) => {
    switch (riskLevel) {
      case 'critical':
        return 'text-red-600 bg-red-50';
      case 'high':
        return 'text-red-500 bg-red-50';
      case 'medium':
        return 'text-yellow-600 bg-yellow-50';
      case 'low':
        return 'text-green-600 bg-green-50';
      default:
        return 'text-gray-600 bg-gray-50';
    }
  };

  const getStatusIcon = (status: string, riskLevel?: string) => {
    switch (status) {
      case 'approved':
        return <CheckCircleIcon className="h-5 w-5 text-green-500" />;
      case 'declined':
        return <XCircleIcon className="h-5 w-5 text-red-500" />;
      case 'flagged':
        return <ExclamationTriangleIcon className="h-5 w-5 text-red-500" />;
      case 'pending':
        return <ClockIcon className="h-5 w-5 text-yellow-500" />;
      default:
        return <ClockIcon className="h-5 w-5 text-gray-500" />;
    }
  };

  const filteredTransactions = transactions.filter(transaction => {
    switch (filter) {
      case 'flagged':
        return transaction.status === 'flagged';
      case 'high_risk':
        return transaction.analysis?.risk_level === 'high' || transaction.analysis?.risk_level === 'critical';
      default:
        return true;
    }
  });

  const togglePause = useCallback(() => {
    setIsPaused(prev => !prev);
  }, []);

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200">
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <h3 className="text-lg font-medium text-gray-900">
              Real-time Transaction Feed
            </h3>
            <div className="flex items-center space-x-2">
              <div className={`h-2 w-2 rounded-full ${isConnected ? 'bg-green-400' : 'bg-red-400'}`} />
              <span className={`text-sm ${isConnected ? 'text-green-600' : 'text-red-600'}`}>
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
          </div>
          
          <div className="flex items-center space-x-3">
            <button
              onClick={togglePause}
              className={`px-3 py-1 rounded-md text-sm font-medium ${
                isPaused 
                  ? 'bg-green-100 text-green-700 hover:bg-green-200' 
                  : 'bg-yellow-100 text-yellow-700 hover:bg-yellow-200'
              }`}
            >
              {isPaused ? 'Resume' : 'Pause'}
            </button>
            
            {showFilters && (
              <select
                value={filter}
                onChange={(e) => setFilter(e.target.value as any)}
                className="text-sm border border-gray-300 rounded-md px-3 py-1 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">All Transactions</option>
                <option value="flagged">Flagged Only</option>
                <option value="high_risk">High Risk Only</option>
              </select>
            )}
          </div>
        </div>
      </div>

      {/* Transaction List */}
      <div className="max-h-96 overflow-y-auto">
        <AnimatePresence>
          {filteredTransactions.map((transaction) => (
            <motion.div
              key={transaction.id}
              initial={{ opacity: 0, y: -20, backgroundColor: '#f0f9ff' }}
              animate={{ 
                opacity: 1, 
                y: 0, 
                backgroundColor: transaction.isNew ? '#f0f9ff' : '#ffffff' 
              }}
              exit={{ opacity: 0, x: -100 }}
              transition={{ duration: 0.3 }}
              className={`px-6 py-4 border-b border-gray-100 hover:bg-gray-50 cursor-pointer ${
                transaction.isNew ? 'ring-2 ring-blue-200' : ''
              }`}
              onClick={() => onTransactionClick?.(transaction)}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  {getStatusIcon(transaction.status, transaction.analysis?.risk_level)}
                  
                  <div>
                    <div className="flex items-center space-x-2">
                      <span className="font-medium text-gray-900">
                        {transaction.transaction_id}
                      </span>
                      {transaction.analysis && (
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                          getRiskColor(transaction.analysis.risk_level)
                        }`}>
                          {transaction.analysis.risk_level?.toUpperCase()}
                        </span>
                      )}
                    </div>
                    
                    <div className="text-sm text-gray-500">
                      {transaction.user_id} • {transaction.merchant_id}
                    </div>
                  </div>
                </div>

                <div className="text-right">
                  <div className="font-medium text-gray-900">
                    {new Intl.NumberFormat('en-US', {
                      style: 'currency',
                      currency: transaction.currency
                    }).format(transaction.amount)}
                  </div>
                  
                  <div className="text-sm text-gray-500">
                    {format(new Date(transaction.timestamp), 'HH:mm:ss')}
                  </div>
                  
                  {transaction.analysis && (
                    <div className="text-xs text-gray-400">
                      Risk: {(transaction.analysis.fraud_probability * 100).toFixed(1)}%
                    </div>
                  )}
                </div>
              </div>

              {/* Fraud Analysis Details */}
              {transaction.analysis && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  className="mt-3 pt-3 border-t border-gray-100"
                >
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-500">ML Models:</span>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {transaction.analysis.ml_results.models_used.map((model) => (
                          <span
                            key={model}
                            className="px-2 py-1 bg-blue-100 text-blue-700 rounded text-xs"
                          >
                            {model}
                          </span>
                        ))}
                      </div>
                    </div>
                    
                    <div>
                      <span className="text-gray-500">Rules Triggered:</span>
                      <div className="text-gray-900 mt-1">
                        {transaction.analysis.rule_results.filter(r => r.triggered).length} / {transaction.analysis.rule_results.length}
                      </div>
                    </div>
                  </div>

                  {transaction.analysis.explanation.key_factors.length > 0 && (
                    <div className="mt-2">
                      <span className="text-gray-500 text-sm">Key Risk Factors:</span>
                      <ul className="mt-1 text-sm text-gray-700">
                        {transaction.analysis.explanation.key_factors.slice(0, 2).map((factor, index) => (
                          <li key={index} className="flex items-center space-x-1">
                            <span className="w-1 h-1 bg-gray-400 rounded-full" />
                            <span>{factor}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </motion.div>
              )}
            </motion.div>
          ))}
        </AnimatePresence>

        {filteredTransactions.length === 0 && (
          <div className="px-6 py-8 text-center text-gray-500">
            <ArrowPathIcon className="h-8 w-8 mx-auto mb-2 animate-spin" />
            <p>Waiting for transactions...</p>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="px-6 py-3 bg-gray-50 border-t border-gray-200">
        <div className="flex items-center justify-between text-sm text-gray-600">
          <span>
            Showing {filteredTransactions.length} of {transactions.length} transactions
          </span>
          <span>
            Last updated: {new Date().toLocaleTimeString()}
          </span>
        </div>
      </div>
    </div>
  );
};

export default RealTimeTransactionFeed;