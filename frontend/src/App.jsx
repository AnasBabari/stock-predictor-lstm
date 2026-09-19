import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { fetchSimpleForecast, fetchTickerNews, wakeForecastService } from './api/simpleForecastClient';
import { fetchPriceHistory } from './api/priceHistoryClient';
import PriceChart from './components/PriceChart';
import VolatilityOutlook from './components/VolatilityOutlook';
import ForecastLedgerTrackRecord from './components/ForecastLedgerTrackRecord';
import { ALL_VALID_TICKERS, ALL_TICKERS_SET } from './universe';

function formatMoney(value, currencySymbol = '$') {
  if (value == null || value === '' || !Number.isFinite(Number(value))) return '—';
  const num = Number(value);
  if (currencySymbol === 'p' || currencySymbol === 'GBp') {
    return `${num.toLocaleString('en-GB', { minimumFractionDigits: 1, maximumFractionDigits: 1 })}p`;
  }
  return num.toLocaleString('en-US', { style: 'currency', currency: 'USD' });
}

function formatPercent(value, digits = 1) {
  if (value == null || value === '' || !Number.isFinite(Number(value))) return '—';
  return `${Number(value).toFixed(digits)}%`;
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
  const hasRatio = backtest.relative_mae_vs_persistence != null && backtest.relative_mae_vs_persistence !== '' && Number.isFinite(Number(backtest.relative_mae_vs_persistence));
  const ratio = hasRatio ? Number(backtest.relative_mae_vs_persistence) : null;
  const beatBaseline = hasRatio && ratio < 1;
  const directionAcc = backtest.direction_accuracy != null && backtest.direction_accuracy !== '' && Number.isFinite(Number(backtest.direction_accuracy)) ? Number(backtest.direction_accuracy) * 100 : null;
  return (
    <section className="panel evidence-panel" aria-label="Historical Model Performance">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Past performance</p>
          <h2>Price Model Historical Performance</h2>
        </div>
        {hasRatio && (
          <span className={`verdict ${beatBaseline ? 'positive' : 'caution'}`}>
            {beatBaseline ? 'More accurate than assuming no price change' : 'No better than assuming no price change'}
          </span>
        )}
      </div>
      <div className="metrics-grid">
        <article>
          <span>Average error</span>
          <strong className="mono">{formatPercent(backtest.mae_percent, 2)}</strong>
          <small className="kpi-subtext">Average size of the prediction mistakes</small>
        </article>
        <article>
          <span>Larger-error score</span>
          <strong className="mono">{formatPercent(backtest.rmse_percent, 2)}</strong>
          <small className="kpi-subtext">Gives bigger mistakes more weight; lower is better</small>
        </article>
        <article>
          <span>Up or down correct</span>
          <strong className="mono">{formatPercent(directionAcc, 1)}</strong>
          <small className="kpi-subtext">How often the price direction was right</small>
        </article>
        <article>
          <span>Compared with no price change</span>
          <strong className="mono">{Number.isFinite(ratio) ? `${ratio.toFixed(2)}×` : '—'}</strong>
          <small className="kpi-subtext">Below 1 means smaller average mistakes</small>
        </article>
      </div>
      <p className="method-note">
        Tested on {backtest.test_samples} past forecasts from {backtest.test_start} to {backtest.test_end}.
        These dates were kept separate when choosing the model. Past results do not guarantee future performance.
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
      label: 'Positive tone',
    };
  }
  if (badge === 'Bearish' || label === 'negative' || label === 'bearish' || score < -0.15) {
    return {
      type: 'bearish',
      label: 'Negative tone',
    };
  }
  return {
    type: 'neutral',
    label: 'Neutral tone',
  };
}

function formatProviderLabel(provider) {
  if (!provider || provider === 'none') return null;
  const p = String(provider).toLowerCase();
  if (p === 'institutional_feed') return 'News feed';
  if (p === 'alpaca') return 'Alpaca';
  if (p === 'yahoo') return 'Yahoo Finance';
  if (p === 'sec_edgar') return 'Company filings';
  return provider.replace(/_/g, ' ');
}

function NewsPanel({ news, ticker, loading }) {
  const [showAll, setShowAll] = useState(false);
  const items = news?.items || [];
  const providerLabel = formatProviderLabel(news?.provider);
  const displayedItems = showAll ? items : items.slice(0, 5);

  return (
    <section className="panel news-panel" aria-label="Live Market Headlines">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">In the news</p>
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
            Not included in the forecast
          </span>
        </div>
      </div>

      <p className="method-note news-top-disclosure">
        These headlines are provided for market context only and do not affect the displayed price forecast.
      </p>

      {loading && !items.length ? (
        <div className="news-loading-state" role="status" aria-label="Loading headlines">
          <div className="news-skeleton-pulse" />
          <span>Loading recent stories…</span>
        </div>
      ) : items.length ? (
        <>
          <ul className="news-simple-list">
            {displayedItems.map((item, index) => {
              const sentiment = resolveSentimentBadge(item);
              const title = item.title || item.headline || 'Market Update';
              const timestamp = formatNewsTimestamp(item.published_at);
              return (
                <li key={`${item.published_at || ''}-${item.id || index}`} className="news-list-item">
                  <div className="news-item-meta">
                    <span className="news-source">{item.source || 'Market Wire'}</span>
                    {timestamp && <time className="news-time" dateTime={item.published_at}>{timestamp}</time>}
                    <span className={`sentiment-badge ${sentiment.type}`}>
                      {sentiment.label}
                    </span>
                  </div>
                  <h3 className="news-item-headline">
                    {item.url ? (
                      <a href={item.url} target="_blank" rel="noreferrer noopener">
                        {title} <span className="external-arrow" aria-hidden="true">↗</span>
                      </a>
                    ) : (
                      title
                    )}
                  </h3>
                </li>
              );
            })}
          </ul>

          {items.length > 5 && (
            <button
              type="button"
              className="show-more-news-btn"
              onClick={() => setShowAll((prev) => !prev)}
            >
              {showAll ? 'Show fewer stories' : `Show more stories (${items.length - 5} more)`}
            </button>
          )}
        </>
      ) : (
        <p className="empty-copy">No recent stories are available right now.</p>
      )}
    </section>
  );
}

export default function App() {
  const [inputTicker, setInputTicker] = useState('');
  const [submittedTicker, setSubmittedTicker] = useState('');
  const [chartTicker, setChartTicker] = useState(null);
  const [activeTab, setActiveTab] = useState('overview'); // 'overview' | 'news' | 'performance'
  const [serviceStatus, setServiceStatus] = useState('checking');
  const [wakeAttempt, setWakeAttempt] = useState(1);
  const [loading, setLoading] = useState(false);
  const [forecast, setForecast] = useState(null);
  const [news, setNews] = useState(null);
  const [newsLoading, setNewsLoading] = useState(false);
  const [inputError, setInputError] = useState('');
  const [historyError, setHistoryError] = useState('');
  const [forecastError, setForecastError] = useState('');
  const [perf, setPerf] = useState(null);

  const wakeController = useRef(null);
  const requestController = useRef(null);
  const requestSeq = useRef(0);
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

  useEffect(() => () => requestController.current?.abort(), []);

  const nowMs = useCallback(() => (typeof performance !== 'undefined' ? performance.now() : Date.now()), []);

  const handleHistorySettled = useCallback((report) => {
    const chartMs = Number.isFinite(report?.fetchMs)
      ? Math.round(report.fetchMs)
      : Math.round(nowMs() - (perfT0.current || nowMs()));
    const cacheLabel = !report?.ok
      ? 'history failed'
      : report.fromCache
        ? 'browser cache'
        : report.degraded
          ? 'forecast payload'
          : report.marketDataCache === 'hit'
            ? 'cache hit'
            : 'cold fetch';
    setPerf((prev) => ({ ...(prev || {}), chartMs, fetchMs: report?.fetchMs ?? null, cacheLabel, degraded: Boolean(report?.degraded) }));
  }, [nowMs]);

  const runForecast = useCallback(async (eventOrSymbol) => {
    if (eventOrSymbol && typeof eventOrSymbol.preventDefault === 'function') {
      eventOrSymbol.preventDefault();
    }
    const raw = typeof eventOrSymbol === 'string' ? eventOrSymbol : inputTicker;
    const symbol = String(raw || '').trim().toUpperCase();
    if (!symbol) return;

    if (!/^[A-Z0-9.\-_]{1,15}$/.test(symbol)) {
      setInputError('Please enter a valid stock ticker symbol (e.g. MSFT, SHEL.L, NVDA, ARM).');
      return;
    }
    if (!ALL_TICKERS_SET.has(symbol)) {
      setInputError(`Choose one of the ${ALL_VALID_TICKERS.length} supported LSE, NASDAQ, and NYSE tickers. This symbol is not supported yet.`);
      return;
    }

    setInputError('');
    setHistoryError('');
    setForecastError('');

    // Cancel outstanding requests and ignore late responses
    requestController.current?.abort();
    const controller = new AbortController();
    requestController.current = controller;
    const seq = ++requestSeq.current;

    // Clear previous results when selecting a different stock
    if (chartTicker !== symbol) {
      setChartTicker(null);
    }
    setForecast(null);
    setNews(null);
    setPerf(null);
    setSubmittedTicker(symbol);
    setLoading(true);
    setNewsLoading(true);

    perfT0.current = (typeof performance !== 'undefined' ? performance.now() : Date.now());

    let historyUsable = chartTicker === symbol;
    let fallbackAvailable = false;

    // 1. History request: reveals chart when usable historical data arrives
    const historyPromise = fetchPriceHistory(symbol, { signal: controller.signal })
      .then((historyResult) => {
        if (requestSeq.current !== seq) return null;
        if (Array.isArray(historyResult?.daily) && historyResult.daily.length > 0) {
          historyUsable = true;
          setChartTicker(symbol);
        }
        return historyResult;
      })
      .catch((err) => {
        if (requestSeq.current !== seq) return null;
        if (err?.name !== 'AbortError') {
          return { error: err };
        }
        return null;
      });

    // 2. Forecast request
    const forecastPromise = (async () => {
      if (serviceStatus !== 'online') {
        await wakeForecastService({ signal: controller.signal, onAttempt: setWakeAttempt });
        if (requestSeq.current !== seq) return null;
        setServiceStatus('online');
      }
      const forecastT0 = nowMs();
      const forecastValue = await fetchSimpleForecast(symbol, { signal: controller.signal });
      if (requestSeq.current !== seq) return null;
      setForecast(forecastValue);
      setPerf((prev) => ({ ...(prev || {}), forecastMs: Math.round(nowMs() - forecastT0) }));

      // Fallback check if primary history failed
      const dates = forecastValue?.historical_dates;
      const prices = forecastValue?.historical_prices;
      if (
        forecastValue?.historical_provenance !== 'synthetic' &&
        Array.isArray(dates) &&
        Array.isArray(prices) &&
        dates.length >= 2
      ) {
        fallbackAvailable = true;
        if (!historyUsable) {
          setChartTicker(symbol);
        }
      }
      return forecastValue;
    })().catch((err) => {
      if (requestSeq.current !== seq) return null;
      if (err?.name !== 'AbortError') {
        setForecastError(err?.message || 'The forecast could not be completed.');
      }
      return null;
    });

    // 3. News request
    const newsPromise = fetchTickerNews(symbol, { signal: controller.signal })
      .then((newsResult) => {
        if (requestSeq.current !== seq) return null;
        setNews(newsResult);
        return newsResult;
      })
      .catch((err) => {
        if (requestSeq.current !== seq) return null;
        if (err?.name !== 'AbortError') {
          setNews({ status: 'unavailable', items: [] });
        }
        return null;
      });

    const [histSettled] = await Promise.allSettled([historyPromise, forecastPromise, newsPromise]);
    if (requestSeq.current !== seq) return;
    setLoading(false);
    setNewsLoading(false);

    if (!historyUsable && !fallbackAvailable) {
      const histErr = histSettled?.value?.error?.message;
      setHistoryError(histErr || 'Price history is unavailable for this stock right now.');
    }
  }, [inputTicker, nowMs, serviceStatus, chartTicker]);

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
            <small>Stock forecasts made simple</small>
          </div>
        </a>

        <div className="topbar-right">
          <ServiceBadge status={serviceStatus} attempt={wakeAttempt} />
        </div>
      </header>

      <main id="top">
        {/* Search Header: Large heading on empty first visit; compact row beneath brand after submission */}
        <section className={`search-section ${submittedTicker ? 'compact-search-section' : 'opening-search-section'}`}>
          {!submittedTicker && (
            <h1 className="explore-heading">Which stock would you like to explore?</h1>
          )}

          <form className="stock-search-form" onSubmit={runForecast} noValidate>
            <div className="search-input-row">
              <div className="input-field-group">
                <label htmlFor="tickerInput" className="sr-only">Stock ticker</label>
                <input
                  id="tickerInput"
                  aria-label="Stock ticker"
                  value={inputTicker}
                  onChange={(event) => {
                    setInputTicker(event.target.value.toUpperCase());
                    if (inputError) setInputError('');
                  }}
                  placeholder="Enter a stock ticker"
                  maxLength={15}
                  autoComplete="off"
                  spellCheck="false"
                  aria-invalid={Boolean(inputError)}
                  aria-describedby={inputError ? 'symbolError' : undefined}
                />
                {inputError && (
                  <span id="symbolError" className="symbol-feedback" role="alert">
                    {inputError}
                  </span>
                )}
              </div>

              <button
                className="view-outlook-btn"
                type="submit"
                disabled={!inputTicker.trim()}
              >
                View outlook
              </button>
            </div>
          </form>

          {loading && (
            <div className="search-progress" role="status" aria-live="polite">
              <span className="progress-dot" aria-hidden="true" />
              <span>
                {serviceStatus !== 'online'
                  ? `Starting forecast service${wakeAttempt > 1 ? ` · attempt ${wakeAttempt}` : ''}… Preparing data for ${submittedTicker}…`
                  : `Loading data for ${submittedTicker}…`}
              </span>
            </div>
          )}

          {historyError && !loading && (
            <div className="compact-error-message" role="alert">
              <span>{historyError}</span>
              <button
                type="button"
                className="retry-action-btn"
                onClick={() => runForecast(submittedTicker)}
              >
                Retry
              </button>
            </div>
          )}
        </section>

        {/* Results Workspace */}
        {chartTicker && (
          <div className="results">
            {/* Centerpiece Interactive Price Chart */}
            <PriceChart
              ticker={chartTicker}
              currencySymbol={
                forecast?.ticker === chartTicker && forecast?.currency_symbol
                  ? forecast.currency_symbol
                  : (chartTicker.endsWith('.L') ? 'p' : '$')
              }
              forecast={forecast?.ticker === chartTicker ? forecast : null}
              companyName={forecast?.ticker === chartTicker ? forecast.ticker_name : null}
              onHistorySettled={handleHistorySettled}
              onRetryForecast={() => runForecast(submittedTicker)}
            />

            {forecastError && !loading && (
              <div className="actionable-forecast-error" role="alert">
                <span>{forecastError}</span>
                <button
                  type="button"
                  className="retry-action-btn"
                  onClick={() => runForecast(submittedTicker)}
                >
                  Retry forecast
                </button>
              </div>
            )}

            {/* Supporting Information Behind Overview / News / Performance Tabs */}
            <div className="supporting-tabs-container">
              <div className="tab-navigation-bar" role="tablist" aria-label="Supporting information tabs"
                onKeyDown={(event) => {
                  const keys = ['ArrowRight', 'ArrowLeft', 'Home', 'End'];
                  if (!keys.includes(event.key)) return;
                  event.preventDefault();
                  const tabs = ['overview', 'news', 'performance'];
                  const index = tabs.indexOf(activeTab);
                  const next = event.key === 'Home' ? 0 : event.key === 'End' ? 2
                    : (index + (event.key === 'ArrowRight' ? 1 : 2)) % 3;
                  setActiveTab(tabs[next]);
                  document.getElementById(`tab-${tabs[next]}`)?.focus();
                }}>
                <button
                  type="button"
                  role="tab"
                  id="tab-overview"
                  tabIndex={activeTab === 'overview' ? 0 : -1}
                  aria-selected={activeTab === 'overview'}
                  aria-controls="panel-overview"
                  className={`sub-tab-btn ${activeTab === 'overview' ? 'active' : ''}`}
                  onClick={() => setActiveTab('overview')}
                >
                  Overview
                </button>
                <button
                  type="button"
                  role="tab"
                  id="tab-news"
                  tabIndex={activeTab === 'news' ? 0 : -1}
                  aria-selected={activeTab === 'news'}
                  aria-controls="panel-news"
                  className={`sub-tab-btn ${activeTab === 'news' ? 'active' : ''}`}
                  onClick={() => setActiveTab('news')}
                >
                  News{news?.items?.length ? ` (${news.items.length})` : ''}
                </button>
                <button
                  type="button"
                  role="tab"
                  id="tab-performance"
                  tabIndex={activeTab === 'performance' ? 0 : -1}
                  aria-selected={activeTab === 'performance'}
                  aria-controls="panel-performance"
                  className={`sub-tab-btn ${activeTab === 'performance' ? 'active' : ''}`}
                  onClick={() => setActiveTab('performance')}
                >
                  Performance
                </button>
              </div>

              {activeTab === 'overview' && (
                <div id="panel-overview" role="tabpanel" aria-labelledby="tab-overview">
                  <VolatilityOutlook
                    ticker={chartTicker}
                    currencySymbol={
                      forecast?.ticker === chartTicker && forecast?.currency_symbol
                        ? forecast.currency_symbol
                        : (chartTicker.endsWith('.L') ? 'p' : '$')
                    }
                    currentPrice={forecast?.ticker === chartTicker ? forecast?.current_price : null}
                  />
                </div>
              )}

              {activeTab === 'news' && (
                <div id="panel-news" role="tabpanel" aria-labelledby="tab-news">
                  <NewsPanel
                    news={news}
                    ticker={chartTicker}
                    loading={newsLoading}
                  />
                </div>
              )}

              {activeTab === 'performance' && (
                <div id="panel-performance" role="tabpanel" aria-labelledby="tab-performance">
                  {forecast?.backtest ? (
                    <BacktestPanel backtest={forecast.backtest} />
                  ) : (
                    <div className="panel empty-performance-panel">
                      <p className="empty-copy">Historical evaluation metrics for the price model will appear when the forecast finishes loading.</p>
                    </div>
                  )}

                  <ForecastLedgerTrackRecord ticker={chartTicker} />

                  {perf && (
                    <details className="perf-technical-details">
                      <summary>Technical & operational diagnostics</summary>
                      <dl className="technical-dl">
                        <div>
                          <dt>Price history load:</dt>
                          <dd className="mono">{perf.chartMs != null ? `${perf.chartMs}ms` : '—'}</dd>
                        </div>
                        <div>
                          <dt>Forecast compute:</dt>
                          <dd className="mono">{perf.forecastMs != null ? `${perf.forecastMs}ms` : '—'}</dd>
                        </div>
                        <div>
                          <dt>Market data cache:</dt>
                          <dd className="mono">{perf.cacheLabel || '—'}</dd>
                        </div>
                      </dl>
                    </details>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      <footer>
        <span>Experimental estimates, not financial advice.</span>
      </footer>
    </div>
  );
}
