import React, { useCallback, useEffect, useRef, useState } from 'react';
import { wakeForecastService } from './api/healthClient';
import { fetchPriceHistory } from './api/priceHistoryClient';
import PriceChart from './components/PriceChart';
import VolatilityOutlook from './components/VolatilityOutlook';
import { ALL_VALID_TICKERS, ALL_TICKERS_SET } from './universe';

function ServiceBadge({ status, attempt }) {
  const copy = {
    checking: `Starting volatility service${attempt > 1 ? ` · attempt ${attempt}` : ''}…`,
    online: 'Volatility engine ready',
    offline: 'Volatility service offline',
  }[status];
  return (
    <div className={`service-badge ${status}`} role="status" aria-label={copy}>
      <span />
      {copy}
    </div>
  );
}

export default function App() {
  const [ticker, setTicker] = useState('MSFT');
  const [chartTicker, setChartTicker] = useState('MSFT');
  const [serviceStatus, setServiceStatus] = useState('checking');
  const [wakeAttempt, setWakeAttempt] = useState(1);
  const [error, setError] = useState('');
  const [perf, setPerf] = useState(null);
  const wakeController = useRef(null);
  const perfT0 = useRef(0);

  const wake = useCallback(async () => {
    wakeController.current?.abort();
    const controller = new AbortController();
    wakeController.current = controller;
    setServiceStatus('checking');
    try {
      await wakeForecastService({ signal: controller.signal, onAttempt: setWakeAttempt });
      setServiceStatus('online');
    } catch (wakeError) {
      if (wakeError?.name !== 'AbortError') setServiceStatus('offline');
    }
  }, []);

  useEffect(() => {
    wake();
    return () => wakeController.current?.abort();
  }, [wake]);

  const nowMs = useCallback(() => (typeof performance !== 'undefined' ? performance.now() : Date.now()), []);

  const handleHistorySettled = useCallback((report) => {
    const chartMs = Math.round(nowMs() - (perfT0.current || nowMs()));
    const cacheLabel = !report?.ok
      ? 'history failed'
      : report.fromCache
        ? 'browser cache'
        : report.marketDataCache === 'hit'
          ? 'cache hit'
          : 'cold fetch';
    setPerf((prev) => ({
      ...(prev || {}),
      chartMs,
      fetchMs: report?.fetchMs ?? null,
      cacheLabel,
    }));
  }, [nowMs]);

  // Warm the backend on first paint with the default ticker
  useEffect(() => {
    const controller = new AbortController();
    fetchPriceHistory('MSFT', { signal: controller.signal }).catch(() => {});
    return () => controller.abort();
  }, []);

  const handleSelectTicker = useCallback((eventOrSymbol) => {
    if (eventOrSymbol && typeof eventOrSymbol.preventDefault === 'function') {
      eventOrSymbol.preventDefault();
    }
    const symbol = (typeof eventOrSymbol === 'string' && eventOrSymbol.trim() ? eventOrSymbol : ticker).trim().toUpperCase();
    if (!symbol || !/^[A-Z0-9.\-_]{1,15}$/.test(symbol)) {
      setError('Please enter a valid stock ticker symbol (e.g. MSFT, SHEL.L, NVDA, AAPL).');
      return;
    }
    if (!ALL_TICKERS_SET.has(symbol)) {
      setError(`Choose one of the ${ALL_VALID_TICKERS.length} supported LSE, NASDAQ, and NYSE tickers.`);
      return;
    }
    setError('');
    setTicker(symbol);
    setChartTicker(symbol);
    perfT0.current = nowMs();
    setPerf(null);
  }, [nowMs, ticker]);

  const currencySymbol = chartTicker.endsWith('.L') ? 'p' : '$';

  return (
    <div className="app-shell">
      <div className="ambient-orbs" aria-hidden="true">
        <div className="orb orb-1" />
        <div className="orb orb-2" />
      </div>

      <header className="topbar">
        <a className="brand" href="#top" aria-label="Signal Seven home">
          <span className="brand-icon">S7</span>
          <div className="brand-title-wrap">
            <b>Signal Seven</b>
            <small>Equity Volatility Engine</small>
          </div>
        </a>

        <div className="topbar-right">
          <span className="market-badge">
            US & UK stocks
          </span>
          <ServiceBadge status={serviceStatus} attempt={wakeAttempt} />
        </div>
      </header>

      <main id="top">
        <section className="hero">
          <div className="hero-intro">
            <p className="eyebrow">Equity Risk & Dispersion Analytics</p>
            <h1>Forecast Stock<br /><em>Volatility & Swings.</em></h1>
            <p className="hero-copy">
              Explore price history and inspect multi-horizon volatility forecasts (5, 10, and 20 sessions)
              with zero-drift expected dispersion cones powered by the G3 GPU-trained XGBoost engine.
            </p>
          </div>

          <form className="forecast-form" onSubmit={handleSelectTicker}>
            <div className="form-heading">
              <strong>Select Stock</strong>
              <span>286 supported US & UK tickers</span>
            </div>
            <label htmlFor="ticker">Stock ticker</label>
            <div className="input-row">
              <input
                id="ticker"
                value={ticker}
                onChange={(event) => setTicker(event.target.value.toUpperCase())}
                placeholder="Search a stock ticker, e.g. MSFT, AAPL, SHEL.L"
                maxLength={15}
                autoComplete="off"
                spellCheck="false"
              />
              <div className="horizon-lock">
                <small>Engine</small>
                <strong>G3 Volatility</strong>
              </div>
            </div>

            <button className="submit-forecast" type="submit">
              Analyze {ticker} Volatility
            </button>
          </form>
          {serviceStatus === 'offline' && (
            <button className="retry-button" onClick={wake} type="button">Try connecting again</button>
          )}
          {error && <div className="error-message" role="alert">{error}</div>}
        </section>

        {chartTicker && (
          <div className="results">
            <PriceChart
              ticker={chartTicker}
              currencySymbol={currencySymbol}
              onHistorySettled={handleHistorySettled}
            />
            {perf?.chartMs != null && (
              <p className="timing-note" role="status">
                Chart {(perf.chartMs / 1000).toFixed(1)}s
                {perf.cacheLabel ? ` · market data: ${perf.cacheLabel}` : ''}
              </p>
            )}

            <div className="chart-legend-strip">
              <div className="legend-pill">
                <span className="legend-color-dot historical-dot" aria-hidden="true" />
                <span>Historical Prices</span>
              </div>
              <div className="legend-pill">
                <span className="legend-color-dot forecast-dot" aria-hidden="true" />
                <span>Expected 5D Volatility Cone (p05–p95)</span>
              </div>
            </div>

            <VolatilityOutlook
              ticker={chartTicker}
            />
          </div>
        )}
      </main>

      <footer>
        <span>Experimental volatility estimates, not financial advice.</span>
      </footer>
    </div>
  );
}
