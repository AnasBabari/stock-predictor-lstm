import React, { useCallback } from 'react';
import SplashScreen from './components/SplashScreen';
import Navbar from './components/Navbar';
import HeroSection from './components/HeroSection';
import SearchCard from './components/SearchCard';
import LoadingIndicator from './components/LoadingIndicator';
import StockInfoGrid from './components/StockInfoGrid';
import StatsBar from './components/StatsBar';
import StockChart from './components/StockChart';
import MetricsCard from './components/MetricsCard';
import HoldoutComparisonChart from './components/HoldoutComparisonChart';
import ModelCard from './components/ModelCard';
import GlobalModelStatus from './components/GlobalModelStatus';
import ForecastChartActions from './components/ForecastChartActions';
import Watchlist from './components/Watchlist';
import PredictionHistory from './components/PredictionHistory';
import ToastContainer from './components/ToastContainer';
import { useTheme } from './hooks/useTheme';
import { useToasts } from './hooks/useToasts';
import { isValidTicker, useWatchlist } from './hooks/useWatchlist';
import { FORECAST_TYPES, useForecast } from './hooks/useForecast';
import { useCompleteAnalysisExport } from './hooks/useCompleteAnalysisExport';

export default function App() {
  const { theme, toggleTheme } = useTheme();
  const { toasts, addToast } = useToasts();

  const {
    ticker,
    setTicker,
    forecastDays,
    setForecastDays,
    daysView,
    setDaysView,
    forecastType,
    setForecastType,
    trainingProfile,
    setTrainingProfile,
    isLoading,
    loadingStage,
    trainingProgress,
    errorMsg,
    setErrorMsg,
    predictionData,
    stockInfo,
    chartRef,
    forecastCacheRef,
    fetchPredictionData,
    handlePredict,
    handleCancelRequest,
    handleClearBrowserModels,
    apiBase,
  } = useForecast({
    addToast,
    onNewTickerSearched: (predictionResult) => addToHistory(predictionResult),
  });

  const {
    watchlist,
    history,
    addToHistory,
    handleAddWatchlist,
    handleRemoveWatchlist,
    handleClearWatchlist,
    handleClearHistory,
  } = useWatchlist({
    addToast,
    forecastType,
    stockInfo,
  });

  const {
    isExportLoading,
    handleExportCompleteAnalysis,
  } = useCompleteAnalysisExport({
    addToast,
    setErrorMsg,
    fetchPredictionData,
    forecastCacheRef,
    ticker,
    forecastDays,
    trainingProfile,
  });

  const handleSelectTicker = useCallback(
    (selectedSymbol) => {
      if (!isValidTicker(selectedSymbol)) {
        addToast('error', 'That history entry has no valid ticker to predict.');
        return;
      }
      const symbol = selectedSymbol.trim().toUpperCase();
      setTicker(symbol);
      handlePredict(symbol);
    },
    [addToast, handlePredict, setTicker]
  );

  const isBusy = isLoading || isExportLoading;

  return (
    <div className="app-container">
      <SplashScreen />
      <Navbar
        theme={theme}
        onToggleTheme={toggleTheme}
        onClearBrowserModels={handleClearBrowserModels}
      />

      <main className="main-content">
        <HeroSection />

        <SearchCard
          ticker={ticker}
          setTicker={setTicker}
          forecastDays={forecastDays}
          setForecastDays={setForecastDays}
          forecastType={forecastType}
          trainingProfile={trainingProfile}
          setTrainingProfile={setTrainingProfile}
          onForecastTypeChange={setForecastType}
          onPredict={handlePredict}
          isLoading={isBusy}
          apiBase={apiBase}
        />

        {isBusy && (
          <LoadingIndicator
            isLoading={isBusy}
            stage={loadingStage}
            progress={trainingProgress}
            profile={trainingProfile}
            onCancel={handleCancelRequest}
          />
        )}

        {errorMsg && (
          <div className="error-banner" role="alert">
            <span className="error-icon">⚠️</span>
            <span className="error-text">{errorMsg}</span>
          </div>
        )}

        {stockInfo && <StockInfoGrid stockInfo={stockInfo} />}

        {predictionData && (
          <>
            <StatsBar
              stockData={predictionData}
              forecastType={forecastType}
              stockInfo={stockInfo}
            />

            <div className="chart-section">
              <div className="chart-header">
                <h2>
                  {predictionData.ticker}{' '}
                  {forecastType === FORECAST_TYPES.TREND ? 'Direction Probability' : 'Price Forecast'}
                </h2>
                <div className="chart-controls">
                  {forecastType === FORECAST_TYPES.PRICE && (
                    <div className="view-selector">
                      <button
                        className={`view-btn ${daysView === 7 ? 'active' : ''}`}
                        onClick={() => setDaysView(7)}
                      >
                        7D
                      </button>
                      <button
                        className={`view-btn ${daysView === 21 ? 'active' : ''}`}
                        onClick={() => setDaysView(21)}
                      >
                        21D
                      </button>
                      <button
                        className={`view-btn ${daysView === 60 ? 'active' : ''}`}
                        onClick={() => setDaysView(60)}
                      >
                        60D
                      </button>
                      <button
                        className={`view-btn ${daysView === 120 ? 'active' : ''}`}
                        onClick={() => setDaysView(120)}
                      >
                        120D
                      </button>
                      <button
                        className={`view-btn ${daysView === 0 ? 'active' : ''}`}
                        onClick={() => setDaysView(0)}
                      >
                        All
                      </button>
                    </div>
                  )}
                </div>
              </div>

              <StockChart
                ref={chartRef}
                stockData={predictionData}
                forecastType={forecastType}
                daysView={daysView}
                setDaysView={setDaysView}
                theme={theme}
              />

              <ForecastChartActions
                chartRef={chartRef}
                stockData={predictionData}
                forecastType={forecastType}
                onAddWatchlist={handleAddWatchlist}
                onToast={addToast}
                onExportCompleteAnalysis={handleExportCompleteAnalysis}
              />
            </div>

            <MetricsCard
              stockData={predictionData}
              forecastType={forecastType}
            />
            <HoldoutComparisonChart data={predictionData} />
            <GlobalModelStatus data={predictionData} />
            <ModelCard data={predictionData} />
          </>
        )}

        <div className="dashboard-grid">
          <Watchlist
            items={watchlist}
            onSelectTicker={handleSelectTicker}
            onRemoveItem={handleRemoveWatchlist}
            onClearAll={handleClearWatchlist}
          />
          <PredictionHistory
            items={history}
            onSelectTicker={handleSelectTicker}
            onClearAll={handleClearHistory}
          />
        </div>
      </main>


      <footer className="footer">
        <p>
          StockLSTM — browser-trained LSTM with purged-holdout evaluation, persistence baselines, and stationary features.
        </p>
      </footer>

      <ToastContainer toasts={toasts} />
    </div>
  );
}
