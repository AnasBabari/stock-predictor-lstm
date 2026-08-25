import JSZip from 'jszip';

import { exportCompleteAnalysis } from './exportService';

function forecast(ticker, days) {
  const dates = Array.from({ length: days }, (_, index) => `2026-08-${index + 1}`);
  return {
    price: {
      ticker,
      forecast_days: days,
      historical_dates: ['2026-07-31'],
      historical_prices: [100],
      future_dates: dates,
      predicted_prices: Array.from({ length: days }, (_, index) => 101 + index),
    },
    direction: {
      ticker,
      forecast_days: days,
      direction_horizon_days: days,
      direction: 'Up',
      direction_probabilities: { down: 0.1, neutral: 0.3, up: 0.6 },
      forecast_status: { state: 'promoted', decision: 'model', alpha: 1, label: 'x' },
      future_dates: dates,
      attention_weights: [{ index: 0, date: '2026-07-31', weight: 1 }],
    },
  };
}

describe('complete analysis export identity', () => {
  beforeEach(() => {
    URL.createObjectURL = vi.fn(() => 'blob:test');
    URL.revokeObjectURL = vi.fn();
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
  });

  it('zips only the selected ticker and horizon after multiple forecasts', async () => {
    const stale = forecast('TSLA', 3);
    const selected = forecast('AAPL', 7);
    expect(stale.price.ticker).not.toBe(selected.price.ticker);

    const metadata = { ticker: 'AAPL', forecast_days: 7, generated_at: '2026-07-26' };
    const { blob, filename } = await exportCompleteAnalysis({
      priceData: selected.price,
      directionData: selected.direction,
      metadata,
    });
    const zip = await JSZip.loadAsync(blob);
    const priceCsv = await zip.file('price_forecast.csv').async('string');
    const directionCsv = await zip.file('direction_forecast.csv').async('string');
    const exportedMetadata = JSON.parse(await zip.file('metadata.json').async('string'));

    expect(filename).toBe('AAPL_complete_analysis.zip');
    expect(exportedMetadata).toEqual(metadata);
    expect(priceCsv.split('\n')).toHaveLength(9);
    // v2 trend CSV: header + 8 key/value rows for the single decision.
    expect(directionCsv.split('\n')).toHaveLength(8);
    expect(directionCsv).toContain('Decision,Up');
    expect(directionCsv).toContain('P(Up),0.600000');
    expect(priceCsv).not.toContain('TSLA');
    expect(directionCsv).not.toContain('TSLA');
  });

  it('rejects mixed ticker or horizon payloads', async () => {
    const aapl = forecast('AAPL', 7);
    const stale = forecast('TSLA', 3);
    await expect(
      exportCompleteAnalysis({
        priceData: aapl.price,
        directionData: stale.direction,
        metadata: { ticker: 'AAPL', forecast_days: 7 },
      })
    ).rejects.toThrow(/identity/);
  });

  it('exports signed volatility evidence without inventing a direction forecast', async () => {
    const days = 3;
    const dates = ['2026-08-04', '2026-08-05', '2026-08-06'];
    const volatility = {
      ticker: 'MSFT',
      forecast_days: days,
      historical_dates: ['2026-07-31'],
      historical_prices: [100],
      future_dates: dates,
      predicted_prices: [100, 100, 100],
      volatility_cone: Object.fromEntries(
        ['p05', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95'].map((key) => [key, [99, 100, 101]]),
      ),
      evidence: { certified: true, metric_source: 'locked_purged_walk_forward' },
      metadata: { engine: { certified_head: 'volatility' } },
    };
    const metadata = {
      ticker: 'MSFT',
      forecast_days: days,
      serving_mode: 'signed_global_volatility',
    };
    const { blob, filename } = await exportCompleteAnalysis({
      priceData: volatility,
      directionData: null,
      metadata,
    });
    const zip = await JSZip.loadAsync(blob);
    expect(filename).toBe('MSFT_volatility_evidence.zip');
    expect(zip.file('volatility_forecast.csv')).not.toBeNull();
    expect(zip.file('market_history.csv')).not.toBeNull();
    expect(zip.file('evidence.json')).not.toBeNull();
    expect(zip.file('direction_forecast.csv')).toBeNull();
  });
});
