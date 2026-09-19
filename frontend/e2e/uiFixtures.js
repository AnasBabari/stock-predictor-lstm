const origin = '2026-09-03';
const futureDates = ['2026-09-04', '2026-09-08', '2026-09-09', '2026-09-10', '2026-09-11', '2026-09-14', '2026-09-15'];
const volatilityFutureDates = [...futureDates, '2026-09-16', '2026-09-17', '2026-09-18',
  '2026-09-21', '2026-09-22', '2026-09-23', '2026-09-24', '2026-09-25',
  '2026-09-28', '2026-09-29', '2026-09-30', '2026-10-01', '2026-10-02'];
const historyDates = ['2026-08-25', '2026-08-26', '2026-08-27', '2026-08-28',
  '2026-08-31', '2026-09-01', '2026-09-02', origin];

export function volatilityPayload(horizon) {
  const dates = volatilityFutureDates.slice(0, horizon);
  const historical = historyDates;
  const quantiles = Object.fromEntries([
    ['p05', 435], ['p10', 440], ['p25', 445], ['p50', 450],
    ['p75', 455], ['p90', 460], ['p95', 465],
  ].map(([key, value]) => [key, Array(horizon).fill(value)]));
  return {
    ticker: 'MSFT', as_of: origin, horizon, current_price: 450,
    historical_dates: historical, historical_prices: historical.map((_, i) => 447.9 + i * 0.3),
    forecast: {
      future_dates: dates,
      price_quantiles: quantiles,
      expected_annualized_volatility: 0.214,
      predicted_volatility: 0.214,
      model: 'gpu_g3', requested_model: 'gpu_g3', baseline: false,
    },
    evidence: {
      model_status: 'learned_model', baseline: false, model_name: 'gpu_g3',
      model_version: 'g3-qlike-base-margin-v2', metric_source: 'validation_panel',
      data_as_of: origin, data_provider: 'local_fixture', risk_level: 'Moderate',
      trailing_annualized_volatility_60d: 0.19, fallback_used: null,
    },
  };
}

export function installFixtures(page, failedHorizon = null, priceOverrides = {}) {
  return page.route('**/*', async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname.endsWith('/health')) return route.fulfill({ json: { status: 'ok' } });
    if (!url.pathname.startsWith('/api/')) return route.continue();
    if (url.pathname.endsWith('/history')) {
      return route.fulfill({ json: {
        ticker: 'MSFT', as_of: origin,
        daily: historyDates.map((d, i) => ({ d, c: 447.9 + i * 0.3 })),
      } });
    }
    if (url.pathname.includes('/volatility/forecast')) {
      const horizon = Number(url.searchParams.get('horizon') || 5);
      if (horizon === failedHorizon) {
        return route.fulfill({ status: 503, json: { detail: 'Fixture horizon unavailable' } });
      }
      return route.fulfill({ json: volatilityPayload(horizon) });
    }
    if (url.pathname.endsWith('/forecast')) {
      return route.fulfill({ json: {
        ticker: 'MSFT', forecast_days: 7, data_as_of: origin, current_price: 450,
        future_dates: futureDates,
        lower_prices: [440, 441, 442, 443, 444, 445, 446],
        upper_prices: [460, 461, 462, 463, 464, 465, 466],
        ...priceOverrides,
      } });
    }
    if (url.pathname.endsWith('/volatility/ledger')) {
      return route.fulfill({ json: { ticker: 'MSFT', entries: [], live_track_record: {}, replay_track_record: {} } });
    }
    if (url.pathname.endsWith('/news')) return route.fulfill({ json: {
      status: 'available', provider: 'verification_fixture', ticker: 'MSFT',
      items: [{ id: 'example-1', title: '[Example fixture] Company update for layout verification',
        source: 'Example fixture — not live news', published_at: '2026-09-03T12:00:00Z' }],
    } });
    return route.fulfill({ json: {} });
  });
}
