import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { fetchSimpleForecast, fetchTickerNews, wakeForecastService } from './api/simpleForecastClient';
import SimpleForecastChart, { midpointPrices } from './components/SimpleForecastChart';
import ForecastLedgerTrackRecord from './components/ForecastLedgerTrackRecord';
import { ALL_VALID_TICKERS, ALL_TICKERS_SET } from './universe';

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
          <p className="eyebrow">Past performance</p>
          <h2>How close were past estimates?</h2>
        </div>
        <span className={`verdict ${beatBaseline ? 'positive' : 'caution'}`}>
          {beatBaseline ? 'More accurate than assuming no price change' : 'No better than assuming no price change'}
        </span>
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
  const items = news?.items || [];
  const providerLabel = formatProviderLabel(news?.provider);

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
      {loading && !items.length ? (
        <div className="news-loading-state" role="status" aria-label="Loading headlines">
          <div className="news-skeleton-pulse" />
          <span>Loading recent stories…</span>
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
                    </span>
                    {item.after_market_close && (
                      <span className="after-hours-tag" title="Published after regular market close">
                        After market close
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
                    <span className="news-wire-note">News update</span>
                  )}
                </div>
              </article>
            );
          })}
        </div>
      ) : (
        <p className="empty-copy">No recent stories are available right now.</p>
      )}
      <p className="method-note">
        These stories help you follow the company, but do not affect this forecast.
        Tone labels describe the wording of a story, not whether you should buy or sell.
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
    if (!symbol || !/^[A-Z0-9.\-_]{1,15}$/.test(symbol)) {
      setError('Please enter a valid stock ticker symbol (e.g. MSFT, SHEL.L, NVDA, ARM).');
      return;
    }
    if (!ALL_TICKERS_SET.has(symbol)) {
      setError(`Choose one of the ${ALL_VALID_TICKERS.length} supported LSE, NASDAQ, and NYSE tickers. This symbol is not supported yet.`);
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
            <small>Stock forecasts made simple</small>
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
            <p className="eyebrow">A clearer view of your stocks</p>
            <h1>Forecast the next<br /><em>7 trading days.</em></h1>
            <p className="hero-copy">
              Choose a stock to see its estimated price over the next seven market days,
              catch up on the news, and see how close past estimates were.
            </p>
            <div className="trust-row">
              <span>US & UK stocks</span>
              <span>Tested on past prices</span>
              <span>Recent company news</span>
            </div>
          </div>

          <form className="forecast-form" onSubmit={runForecast}>
            <div className="form-heading">
              <strong>Create forecast</strong>
              <span>First visit may take a little longer</span>
            </div>
            <label htmlFor="ticker">Stock ticker</label>
            <div className="input-row">
              <input
                id="ticker"
                value={ticker}
                onChange={(event) => setTicker(event.target.value.toUpperCase())}
                placeholder="Search a stock ticker, e.g. MSFT or SHEL.L"
                maxLength={15}
                autoComplete="off"
                spellCheck="false"
              />
              <div className="horizon-lock">
                <small>Looking ahead</small>
                <strong>7 trading days</strong>
              </div>
            </div>

            <button className="submit-forecast" type="submit" disabled={loading}>
              {loading ? 'Preparing your forecast…' : `Run 7-day forecast for ${ticker}`}
            </button>
          </form>
          {serviceStatus === 'offline' && (
            <button className="retry-button" onClick={wake} type="button">Try connecting again</button>
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
                  <span className="model-pill">Estimate, not a guarantee</span>
                </div>
              </div>
              <div className="summary-grid">
                <article>
                  <span>Latest price</span>
                  <strong className="mono">{formatMoney(forecast.current_price, forecast.currency_symbol)}</strong>
                  <small className="kpi-subtext">At the last market close</small>
                </article>
                <article>
                  <span>Estimated price on day 7</span>
                  <strong className="mono">{formatMoney(summary.finalPrice, forecast.currency_symbol)}</strong>
                  <small className="kpi-subtext">Middle of the estimated price range</small>
                </article>
                <article>
                  <span>Estimated change</span>
                  <strong className={`mono ${summary.change >= 0 ? 'up' : 'down'}`}>
                    {summary.change >= 0 ? '+' : ''}{formatPercent(summary.change, 2)}
                  </strong>
                  <small className="kpi-subtext">Compared with the latest price</small>
                </article>
                <article>
                  <span>Time ahead</span>
                  <strong className="mono horizon-kpi">
                    7 Trading Days
                  </strong>
                  <small className="kpi-subtext">Excludes weekends and market holidays</small>
                </article>
              </div>
              <SimpleForecastChart forecast={forecast} />
              <div className="chart-legend-strip">
                <div className="legend-pill">
                  <span className="legend-color-dot historical-dot" aria-hidden="true" />
                  <span>Past prices</span>
                </div>
                <div className="legend-pill">
                  <span className="legend-color-dot forecast-dot" aria-hidden="true" />
                  <span>Average 7-Day Estimate</span>
                </div>
              </div>
              <p className="chart-caption">
                The blue line shows the middle of each day's estimated price range.
                Actual prices can be higher or lower. This is not a guaranteed return.
              </p>
              <details className="forecast-details">
                <summary>How this estimate is made</summary>
                <p>Each point averages the lower and upper price estimates. It is not a probability-weighted average.</p>
                <p>Model: {forecast.model?.name?.replaceAll('_', ' ') || 'Not provided'}.</p>
              </details>
            </section>
            <BacktestPanel backtest={forecast.backtest} />
            <NewsPanel news={news} ticker={forecast.ticker} loading={newsLoading} />
            <ForecastLedgerTrackRecord ticker={forecast.ticker} />
          </div>
        )}
      </main>

      <footer>
        <span>Experimental estimates, not financial advice.</span>
      </footer>
    </div>
  );
}
