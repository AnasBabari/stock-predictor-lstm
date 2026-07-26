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
      future_dates: dates,
      directions: Array.from({ length: days }, () => 'Up'),
      probabilities: Array.from({ length: days }, () => 0.6),
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
    expect(directionCsv.split('\n')).toHaveLength(8);
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
});
