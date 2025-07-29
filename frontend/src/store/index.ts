/**
 * Redux store configuration with RTK Query
 * Production-ready state management for fraud analytics dashboard
 */

import { configureStore } from '@reduxjs/toolkit';
import { setupListeners } from '@reduxjs/toolkit/query';
import { fraudAnalyticsApi } from './api/fraudAnalyticsApi';
import authSlice from './slices/authSlice';
import dashboardSlice from './slices/dashboardSlice';
import transactionsSlice from './slices/transactionsSlice';
import casesSlice from './slices/casesSlice';
import alertsSlice from './slices/alertsSlice';
import settingsSlice from './slices/settingsSlice';

export const store = configureStore({
  reducer: {
    // RTK Query API
    [fraudAnalyticsApi.reducerPath]: fraudAnalyticsApi.reducer,
    
    // Feature slices
    auth: authSlice,
    dashboard: dashboardSlice,
    transactions: transactionsSlice,
    cases: casesSlice,
    alerts: alertsSlice,
    settings: settingsSlice,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: [fraudAnalyticsApi.util.getRunningQueriesThunk.fulfilled.type],
      },
    }).concat(fraudAnalyticsApi.middleware),
  devTools: process.env.NODE_ENV !== 'production',
});

// Enable listener behavior for the store
setupListeners(store.dispatch);

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

// Export hooks
export { useAppDispatch, useAppSelector } from './hooks';