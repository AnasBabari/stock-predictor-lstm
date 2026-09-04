import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { fetchSimpleForecast, fetchTickerNews, wakeForecastService } from './api/simpleForecastClient';
import SimpleForecastChart, { midpointPrices } from './components/SimpleForecastChart';
import ForecastLedgerTrackRecord from './components/ForecastLedgerTrackRecord';

export const TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA', 'SHEL.L', 'AZN.L', 'HSBA.L'];

export const EXCHANGES = [
  {
    id: 'US',
    name: 'US (NASDAQ / NYSE)',
    mic: 'XNAS',
    tickers: ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA'],
  },
  {
    id: 'UK',
    name: 'UK (LSE)',
    mic: 'XLON',
    tickers: ['SHEL.L', 'AZN.L', 'HSBA.L'],
  },
];

function formatMoney(value, currencySymbol = '$') {
  if (!Number.isFinite(Number(value))) return '—';
  const num = Number(value);
  if (currencySymbol === 'p' || currencySymbol === 'GBp') {
    return `${num.toLocaleString('en-GB', { minimumFractionDigits: 1, maximumFractionDigits: 1 })}p`;
  }
  return num.toLocaleString('en-US', { style: 'currency', currency: 'USD' });
}

function formatPercent(value, digits = 1) {
  return Number.isFinite(Number(value)) ? `${Number(value).toFixed(digits)}%` : '—';
}

function ServiceBadge({ status, attempt }) {
  const copy = {
    checking: `Starting forecast service${attempt > 1 ? ` · attempt ${attempt}` : ''}…`,
    online: 'Forecast service ready',
    offline: 'Forecast service unavailable',
  }[status];
  return (
    <div className={`service-badge ${status}`} role="status" aria-label={copy}>
      <span />
      {copy}
    </div>
  );
}

function BacktestPanel({ backtest }) {
  if (!backtest) return null;
  const ratio = Number(backtest.relative_mae_vs_persistence);
  const beatBaseline = ratio < 1;
  const directionAcc = backtest.direction_accuracy != null ? backtest.direction_accuracy * 100 : null;
  return (
    <section className="panel evidence-panel" aria-label="Historical Model Performance">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Historical reality check</p>
          <h2>How this model did before today</h2>
        </div>
        <span className={`verdict ${beatBaseline ? 'positive' : 'caution'}`}>
          {beatBaseline ? 'Beat no-change benchmark' : 'Did not beat no-change benchmark'}
        </span>
      </div>
      <div className="metrics-grid">
        <article>
          <span>Average error</span>
          <strong className="mono">{formatPercent(backtest.mae_percent, 2)}</strong>
          <small className="kpi-subtext">Mean absolute % error</small>
        </article>
        <article>
          <span>RMSE</span>
          <strong className="mono">{formatPercent(backtest.rmse_percent, 2)}</strong>
          <small className="kpi-subtext">Root mean squared error</small>
        </article>
        <article>
          <span>Direction correct</span>
          <strong className="mono">{formatPercent(directionAcc, 1)}</strong>
          <small className="kpi-subtext">Sign hit rate vs baseline</small>
        </article>
        <article>
          <span>Versus no-change</span>
          <strong className="mono">{Number.isFinite(ratio) ? `${ratio.toFixed(2)}×` : '—'}</strong>
          <small className="kpi-subtext">{beatBaseline ? 'Outperformed persistence' : 'Underperformed persistence'}</small>
        </article>
      </div>
      <p className="method-note">
        Trained, selected, and tested in time order using a 70/15/15 split. The test period
        {` ${backtest.test_start} to ${backtest.test_end}`} was not used to choose the model
        ({backtest.test_samples} forecast origins).
      </p>
    </section>
  );
}

function formatNewsTimestamp(publishedAt) {
  if (!publishedAt) return '';
  try {
    const d = new Date(publishedAt);
    if (Number.isNaN(d.getTime())) return String(publishedAt).slice(0, 10);
    return (
      d.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
      }) +
      ' · ' +
      d.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        hour12: false,
      }) +
      ' UTC'
    );
  } catch {
    return String(publishedAt).slice(0, 10);
  }
}

function resolveSentimentBadge(item) {
  const label = String(item.sentiment_label || '').toLowerCase();
  const badge = item.sentiment_badge;
  const score = item.sentiment;
  if (badge === 'Bullish' || label === 'positive' || label === 'bullish' || score > 0.15) {
    return {
      type: 'bullish',
      label: 'Bullish',
      scoreText: Number.isFinite(score) ? (score > 0 ? `+${score.toFixed(2)}` : score.toFixed(2)) : '',
    };
  }
  if (badge === 'Bearish' || label === 'negative' || label === 'bearish' || score < -0.15) {
    return {
      type: 'bearish',
      label: 'Bearish',
      scoreText: Number.isFinite(score) ? score.toFixed(2) : '',
    };
  }
  return {
    type: 'neutral',
    label: 'Neutral',
    scoreText: Number.isFinite(score) ? score.toFixed(2) : '',
  };
}

function formatProviderLabel(provider) {
  if (!provider || provider === 'none') return null;
  const p = String(provider).toLowerCase();
  if (p === 'institutional_feed') return 'Institutional Wire';
  if (p === 'alpaca') return 'Alpaca Feed';
  if (p === 'yahoo') return 'Yahoo Finance';
  if (p === 'sec_edgar') return 'SEC EDGAR Wire';
  return provider.replace(/_/g, ' ');
}

function NewsPanel({ news, ticker, loading }) {
  const items = news?.items || [];
  const providerLabel = formatProviderLabel(news?.provider);

  return (
    <section className="panel news-panel" aria-label="Live Market Headlines">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Institutional Market Context</p>
          <h2>Recent {ticker} headlines</h2>
        </div>
        <div className="heading-badges">
          {providerLabel && (
            <span className="source-provider-pill">
              <span className="provider-dot" aria-hidden="true" />
              {providerLabel}
            </span>
          )}
          <span className="context-label">
            <span className="context-dot" aria-hidden="true" />
            Not used by model yet
          </span>
        </div>
      </div>
      {loading && !items.length ? (
        <div className="news-loading-state" role="status" aria-label="Loading headlines">
          <div className="news-skeleton-pulse" />
          <span>Retrieving verified headlines and sentiment analysis…</span>
        </div>
      ) : items.length ? (
        <div className="news-cards-grid">
          {items.slice(0, 6).map((item, index) => {
            const sentiment = resolveSentimentBadge(item);
            const title = item.title || item.headline || 'Market Update';
            const timestamp = formatNewsTimestamp(item.published_at);
            return (
              <article key={`${item.published_at || ''}-${item.id || index}`} className="news-card">
                <div className="news-card-header">
                  <span className="news-source-tag">{item.source || 'Market Wire'}</span>
                  <div className="news-card-meta">
                    <span className={`sentiment-badge ${sentiment.type}`}>
                      <span className="sentiment-indicator-dot" />
                      {sentiment.label}
                      {sentiment.scoreText ? ` · ${sentiment.scoreText}` : ''}
                    </span>
                    {item.after_market_close && (
                      <span className="after-hours-tag" title="Published after regular market close">
                        After-Hours
                      </span>
                    )}
                    {timestamp && (
                      <time className="news-timestamp" dateTime={item.published_at}>
                        {timestamp}
                      </time>
                    )}
                  </div>
                </div>
                <h3 className="news-card-title">
                  {item.url ? (
                    <a href={item.url} target="_blank" rel="noreferrer noopener">
                      {title}
                    </a>
                  ) : (
                    title
                  )}
                </h3>
                {item.summary && item.summary !== title && (
                  <p className="news-card-summary">{item.summary}</p>
                )}
                <div className="news-card-footer">
                  {item.url ? (
                    <a
                      className="news-external-link"
                      href={item.url}
                      target="_blank"
                      rel="noreferrer noopener"
                    >
                      Read full story <span aria-hidden="true">↗</span>
                    </a>
                  ) : (
                    <span className="news-wire-note">Market wire release</span>
                  )}
                </div>
              </article>
            );
          })}
        </div>
      ) : (
        <p className="empty-copy">No recent headlines are available from the market-data feed.</p>
      )}
      <p className="method-note">
        News is shown as context only. We will add it to training after collecting matching
        timestamped historical headlines and proving that it improves the same held-out test.
      </p>
    </section>
  );
}

export default function App() {
  const [ticker, setTicker] = useState('MSFT');
  const [serviceStatus, setServiceStatus] = useState('checking');
  const [wakeAttempt, setWakeAttempt] = useState(1);
  const [loading, setLoading] = useState(false);
  const [forecast, setForecast] = useState(null);
  const [news, setNews] = useState(null);
  const [newsLoading, setNewsLoading] = useState(false);
  const [error, setError] = useState('');
  const wakeController = useRef(null);
  const requestController = useRef(null);

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

  useEffect(() => () => requestController.current?.abort(), []);

  const runForecast = useCallback(async (eventOrSymbol) => {
    if (eventOrSymbol && typeof eventOrSymbol.preventDefault === 'function') {
      eventOrSymbol.preventDefault();
    }
    const symbol = (typeof eventOrSymbol === 'string' && eventOrSymbol.trim() ? eventOrSymbol : ticker).trim().toUpperCase();
    if (!TICKERS.includes(symbol)) {
      setError(`Choose one of ${TICKERS.join(', ')} for this benchmark.`);
      return;
    }
    requestController.current?.abort();
    const controller = new AbortController();
    requestController.current = controller;
    setTicker(symbol);
    setLoading(true);
    setNewsLoading(true);
    setError('');
    setNews(null);
    try {
      if (serviceStatus !== 'online') {
        await wakeForecastService({ signal: controller.signal, onAttempt: setWakeAttempt });
        setServiceStatus('online');
      }
      const [forecastResult, newsResult] = await Promise.allSettled([
        fetchSimpleForecast(symbol, { signal: controller.signal }),
        fetchTickerNews(symbol, { signal: controller.signal }),
      ]);

      if (forecastResult.status === 'fulfilled') {
        setForecast(forecastResult.value);
      } else {
        throw forecastResult.reason;
      }

      if (newsResult.status === 'fulfilled') {
        setNews(newsResult.value);
      } else {
        setNews({ status: 'unavailable', items: [] });
      }
    } catch (forecastError) {
      if (forecastError?.name !== 'AbortError') {
        setError(forecastError?.message || 'The forecast could not be completed.');
      }
    } finally {
      setLoading(false);
      setNewsLoading(false);
    }
  }, [serviceStatus, ticker]);

  const summary = useMemo(() => {
    if (!forecast?.lower_prices?.length || !forecast?.upper_prices?.length) return null;
    const averagePrices = midpointPrices(forecast.lower_prices, forecast.upper_prices);
    const finalPrice = Number(averagePrices.at(-1));
    const currentPrice = Number(forecast.current_price);
    if (!Number.isFinite(finalPrice) || !Number.isFinite(currentPrice) || currentPrice === 0) return null;
    return { finalPrice, change: ((finalPrice / currentPrice) - 1) * 100 };
  }, [forecast]);

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
            <small>Quantitative Terminal</small>
          </div>
        </a>

        <div className="topbar-center" aria-label="Quick ticker switcher">
          <span className="topbar-center-label">Universe</span>
          {EXCHANGES.map((ex) => (
            <div key={ex.id} className="exchange-group">
              <span className="exchange-tag-mini">{ex.id}</span>
              {ex.tickers.map((symbol) => (
                <button
                  key={`quick-${symbol}`}
                  type="button"
                  className={`fast-ticker-btn ${ticker === symbol ? 'active' : ''}`}
                  onClick={() => runForecast(symbol)}
                >
                  {symbol}
                </button>
              ))}
            </div>
          ))}
        </div>

        <div className="topbar-right">
          <span className="market-badge">
            {forecast?.exchange_mic
              ? `${forecast.exchange_mic} · ${forecast.currency_symbol || '$'}`
              : 'MULTI-EXCHANGE · CAUSAL'}
          </span>
          <ServiceBadge status={serviceStatus} attempt={wakeAttempt} />
        </div>
      </header>

      <main id="top">
        <section className="hero">
          <div className="hero-intro">
            <p className="eyebrow">Multi-Exchange Machine Learning · Causal Market Evidence</p>
            <h1>Forecast the next<br /><em>7 trading days.</em></h1>
            <p className="hero-copy">
              Quantitative multi-exchange forecasting across US (NASDAQ / NYSE) and UK (London Stock Exchange) equities.
              Every estimate comes with the historical holdout test result that proves how the model actually performed before today.
            </p>
            <div className="trust-row">
              <span>70/15/15 time split</span>
              <span>Completed bars only</span>
              <span>LSE & NYSE calendars</span>
              <span>PostgreSQL forward ledger</span>
            </div>
          </div>

          <form className="forecast-form" onSubmit={runForecast}>
            <div className="form-heading">
              <strong>Create forecast</strong>
              <span>Usually 5–15 seconds</span>
            </div>
            <label htmlFor="ticker">Stock ticker</label>
            <div className="input-row">
              <input
                id="ticker"
                value={ticker}
                onChange={(event) => setTicker(event.target.value.toUpperCase())}
                placeholder="Enter a ticker (e.g. MSFT, SHEL.L)"
                maxLength={12}
                autoComplete="off"
                spellCheck="false"
              />
              <div className="horizon-lock">
                <small>Horizon</small>
                <strong>7 trading days</strong>
              </div>
            </div>
            <div className="ticker-row multi-exchange-tickers" aria-label="Supported tickers">
              {EXCHANGES.map((ex) => (
                <div key={`form-ex-${ex.id}`} className="form-exchange-group">
                  <span className="form-exchange-tag">{ex.id}</span>
                  {ex.tickers.map((symbol) => (
                    <button
                      key={symbol}
                      type="button"
                      className={ticker === symbol ? 'active' : ''}
                      onClick={() => runForecast(symbol)}
                    >
                      {symbol}
                    </button>
                  ))}
                </div>
              ))}
            </div>
            <button className="submit-forecast" type="submit" disabled={loading}>
              {loading ? 'Training on history…' : 'Run 7-day forecast'}
            </button>
          </form>
          {serviceStatus === 'offline' && (
            <button className="retry-button" onClick={wake} type="button">Retry starting Render</button>
          )}
          {error && <div className="error-message" role="alert">{error}</div>}
        </section>

        {forecast && summary && (
          <div className="results">
            <section className="panel forecast-panel">
              <div className="panel-heading">
                <div>
                  <p className="eyebrow">
                    {forecast.ticker_name ? `${forecast.ticker_name} (${forecast.ticker})` : forecast.ticker} · {forecast.exchange_name || 'Market'} · data through {forecast.data_as_of}
                  </p>
                  <h2>Average seven-day price estimate</h2>
                </div>
                <div className="heading-badges">
                  <span className="corridor-pill">
                    80% Empirical Corridor
                  </span>
                  {forecast.exchange_mic && (
                    <span className="exchange-pill">{forecast.exchange_mic}</span>
                  )}
                  <span className="model-pill">{forecast.model?.name?.replace('_', ' ')}</span>
                </div>
              </div>
              <div className="summary-grid">
                <article>
                  <span>Current Close</span>
                  <strong className="mono">{formatMoney(forecast.current_price, forecast.currency_symbol)}</strong>
                  <small className="kpi-subtext">Latest trading close</small>
                </article>
                <article>
                  <span>Target Estimate</span>
                  <strong className="mono">{formatMoney(summary.finalPrice, forecast.currency_symbol)}</strong>
                  <small className="kpi-subtext">Projected cone midpoint</small>
                </article>
                <article>
                  <span>Expected Move</span>
                  <strong className={`mono ${summary.change >= 0 ? 'up' : 'down'}`}>
                    {summary.change >= 0 ? '+' : ''}{formatPercent(summary.change, 2)}
                  </strong>
                  <small className="kpi-subtext">Cumulative return</small>
                </article>
                <article>
                  <span>Horizon</span>
                  <strong className="mono horizon-kpi">
                    7 Trading Days
                  </strong>
                  <small className="kpi-subtext">{forecast.provenance?.calendar || 'Trading sessions'}</small>
                </article>
              </div>
              <SimpleForecastChart forecast={forecast} />
              <div className="chart-legend-strip">
                <div className="legend-pill">
                  <span className="legend-color-dot historical-dot" aria-hidden="true" />
                  <span>Historical Close</span>
                </div>
                <div className="legend-pill">
                  <span className="legend-color-dot forecast-dot" aria-hidden="true" />
                  <span>Average 7-Day Estimate</span>
                </div>
                <div className="legend-pill">
                  <span className="legend-color-dot corridor-dot" aria-hidden="true" />
                  <span>80% Empirical Corridor</span>
                </div>
              </div>
              <p className="chart-caption">
                The blue line plots the midpoint between the model's lower and upper estimates for each day.
              </p>
            </section>
            <BacktestPanel backtest={forecast.backtest} />
            <NewsPanel news={news} ticker={forecast.ticker} loading={newsLoading} />
            <ForecastLedgerTrackRecord ticker={forecast.ticker} />
          </div>
        )}
      </main>

      <footer>
        <span>Experimental estimates, not financial advice.</span>
        <span>Completed daily bars only · no intraday leakage · PostgreSQL durable ledger</span>
      </footer>
    </div>
  );
}
