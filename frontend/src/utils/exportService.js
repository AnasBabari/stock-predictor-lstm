import JSZip from 'jszip';

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.download = filename;
  link.href = url;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 500);
}

function csvCell(value) {
  let text = String(value ?? '');
  if (/^[=+\-@]/.test(text)) text = `'${text}`;
  if (/[",\r\n]/.test(text)) text = `"${text.replaceAll('"', '""')}"`;
  return text;
}

function csvFromRows(rows) {
  return rows.map((row) => row.map(csvCell).join(',')).join('\n');
}

function assertCompleteIdentity(priceData, directionData, metadata) {
  const expectedTicker = String(metadata?.ticker || '').toUpperCase();
  const expectedDays = Number(metadata?.forecast_days);
  for (const [name, data] of [
    ['price', priceData],
    ['direction', directionData],
  ]) {
    if (data?.ticker !== expectedTicker || Number(data?.forecast_days) !== expectedDays) {
      throw new Error(`${name} forecast identity does not match the requested export.`);
    }
    if (data.future_dates?.length !== expectedDays) {
      throw new Error(`${name} forecast is incomplete for the requested export.`);
    }
  }
  if (priceData.predicted_prices?.length !== expectedDays) {
    throw new Error('Forecast payload lengths do not match the requested export.');
  }
  if (
    directionData &&
    (directionData.direction_horizon_days !== expectedDays ||
      directionData.direction_probabilities == null)
  ) {
    throw new Error('Direction forecast is not in the v2 three-way contract for the requested export.');
  }
}

export async function exportPriceCSV(stockData) {
  if (!stockData) return;

  const certifiedHead = stockData.metadata?.engine?.certified_head;
  const isVolatility = stockData.metadata?.engine?.volatility_forecast === true
    || certifiedHead === 'volatility'
    || certifiedHead === 'return_distribution'
    || stockData.volatility_cone != null;
  if (isVolatility) {
    const keys = ['p05', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95'];
    const rows = [['Date', 'P05', 'P10', 'P25', 'P50', 'P75', 'P90', 'P95']];
    stockData.future_dates.forEach((date, index) => {
      rows.push([date, ...keys.map((key) => stockData.volatility_cone?.[key]?.[index])]);
    });
    downloadBlob(
      new Blob([csvFromRows(rows)], { type: 'text/csv' }),
      `${stockData.ticker}_${certifiedHead === 'return_distribution' ? 'return_distribution' : 'volatility'}_forecast.csv`,
    );
    return;
  }

  const rows = [['Date', 'Price', 'Type']];
  stockData.historical_dates.forEach((dt, i) => {
    rows.push([dt, stockData.historical_prices[i].toFixed(2), 'Historical']);
  });
  stockData.future_dates.forEach((dt, i) => {
    rows.push([dt, stockData.predicted_prices[i].toFixed(2), 'Predicted']);
  });

  downloadBlob(new Blob([csvFromRows(rows)], { type: 'text/csv' }), `${stockData.ticker}_forecast.csv`);
}

export async function exportTrendCSV(stockData) {
  if (!stockData) return;

  // Direction v2: ONE cumulative-horizon decision, not one per future date.
  const probs = stockData.direction_probabilities || {};
  const rows = [
    ['Field', 'Value'],
    ['Ticker', stockData.ticker],
    ['Direction horizon (trading days)', stockData.direction_horizon_days ?? stockData.forecast_days ?? ''],
    ['Decision', stockData.direction || '—'],
    ['P(Down)', probs.down != null ? Number(probs.down).toFixed(6) : '—'],
    ['P(Neutral)', probs.neutral != null ? Number(probs.neutral).toFixed(6) : '—'],
    ['P(Up)', probs.up != null ? Number(probs.up).toFixed(6) : '—'],
    ['Status', stockData.forecast_status?.state || '—'],
  ];

  downloadBlob(new Blob([csvFromRows(rows)], { type: 'text/csv' }), `${stockData.ticker}_trend.csv`);
}

export async function exportAttentionCSV(stockData) {
  if (!stockData) return;

  const rows = [['Index', 'Date', 'Weight']];
  (stockData.attention_weights || []).forEach((item) => {
    rows.push([item.index, item.date, item.weight.toFixed(6)]);
  });

  downloadBlob(
    new Blob([csvFromRows(rows)], { type: 'text/csv' }),
    `${stockData.ticker}_attention_weights.csv`
  );
}

export async function exportCompleteAnalysis({ priceData, directionData, metadata }) {
  if (metadata?.serving_mode === 'signed_global_volatility' || metadata?.serving_mode === 'causal_volatility_baseline') {
    if (!priceData || priceData.ticker !== String(metadata.ticker).toUpperCase()) {
      throw new Error('Volatility evidence identity does not match the requested export.');
    }
    if (Number(priceData.forecast_days) !== Number(metadata.forecast_days)) {
      throw new Error('Volatility evidence horizon does not match the requested export.');
    }
    const certifiedHead = priceData.metadata?.engine?.certified_head;
    const isVolatility = priceData.metadata?.engine?.volatility_forecast === true
      || certifiedHead === 'volatility'
      || certifiedHead === 'return_distribution'
      || priceData.volatility_cone != null;
    if (!isVolatility) {
      throw new Error('The export is not a volatility forecast payload.');
    }
    const expectedDays = Number(metadata.forecast_days);
    const quantileKeys = ['p05', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95'];
    if (
      !Array.isArray(priceData.future_dates) ||
      priceData.future_dates.length !== expectedDays ||
      !Array.isArray(priceData.historical_dates) ||
      !Array.isArray(priceData.historical_prices) ||
      priceData.historical_dates.length !== priceData.historical_prices.length
    ) {
      throw new Error('Volatility evidence contains misaligned date and price paths.');
    }
    const quantiles = priceData.volatility_cone || {};
    if (quantileKeys.some((key) => (
      !Array.isArray(quantiles[key]) ||
      quantiles[key].length !== expectedDays ||
      quantiles[key].some((value) => !Number.isFinite(Number(value)) || Number(value) <= 0)
    ))) {
      throw new Error('Volatility evidence contains incomplete or invalid quantile paths.');
    }
    const zip = new JSZip();
    const volatilityRows = [['Date', 'P05', 'P10', 'P25', 'P50', 'P75', 'P90', 'P95']];
        priceData.future_dates.forEach((date, index) => {
      volatilityRows.push([
        date,
        quantiles.p05?.[index],
        quantiles.p10?.[index],
        quantiles.p25?.[index],
        quantiles.p50?.[index],
        quantiles.p75?.[index],
        quantiles.p90?.[index],
        quantiles.p95?.[index],
      ]);
    });
    const historyRows = [['Date', 'Close', 'Type']];
    priceData.historical_dates.forEach((date, index) => {
      historyRows.push([date, priceData.historical_prices[index], 'Historical']);
    });
    zip.file('volatility_forecast.csv', csvFromRows(volatilityRows));
    if (certifiedHead === 'return_distribution') {
      const medianRows = [['Date', 'Median Price', 'Type']];
        priceData.future_dates.forEach((date, index) => {
          medianRows.push([date, quantiles.p50?.[index], 'Learned median']);
      });
      zip.file('median_price_forecast.csv', csvFromRows(medianRows));
    }
    zip.file('market_history.csv', csvFromRows(historyRows));
    zip.file('metadata.json', JSON.stringify(metadata, null, 2));
    zip.file('evidence.json', JSON.stringify(priceData.evidence || {}, null, 2));
    const blob = await zip.generateAsync({ type: 'blob' });
    const filename = metadata.serving_mode === 'signed_global_volatility'
      ? `${metadata.ticker}_volatility_evidence.zip`
      : `${metadata.ticker}_volatility_forecast.zip`;
    downloadBlob(blob, filename);
    return { blob, filename };
  }
  assertCompleteIdentity(priceData, directionData, metadata);
  const zip = new JSZip();

  const priceRows = [['Date', 'Price', 'Type']];
  priceData.historical_dates.forEach((dt, i) => {
    priceRows.push([dt, priceData.historical_prices[i].toFixed(2), 'Historical']);
  });
  priceData.future_dates.forEach((dt, i) => {
    priceRows.push([dt, priceData.predicted_prices[i].toFixed(2), 'Predicted']);
  });

  // Direction v2: one cumulative-horizon decision block, mirroring
  // exportTrendCSV (no fabricated per-date decisions).
  const trendProbs = directionData.direction_probabilities || {};
  const directionRows = [
    ['Field', 'Value'],
    ['Ticker', directionData.ticker],
    ['Direction horizon (trading days)', directionData.direction_horizon_days ?? directionData.forecast_days ?? ''],
    ['Decision', directionData.direction || '—'],
    ['P(Down)', trendProbs.down != null ? Number(trendProbs.down).toFixed(6) : '—'],
    ['P(Neutral)', trendProbs.neutral != null ? Number(trendProbs.neutral).toFixed(6) : '—'],
    ['P(Up)', trendProbs.up != null ? Number(trendProbs.up).toFixed(6) : '—'],
    ['Status', directionData.forecast_status?.state || '—'],
  ];

  const attentionRows = [['Index', 'Date', 'Weight']];
  (directionData.attention_weights || []).forEach((item) => {
    attentionRows.push([item.index, item.date, item.weight.toFixed(6)]);
  });

  zip.file('price_forecast.csv', csvFromRows(priceRows));
  zip.file('direction_forecast.csv', csvFromRows(directionRows));
  zip.file('attention_weights.csv', csvFromRows(attentionRows));
  zip.file('metadata.json', JSON.stringify(metadata, null, 2));

  const blob = await zip.generateAsync({ type: 'blob' });
  const filename = `${metadata.ticker}_complete_analysis.zip`;
  downloadBlob(blob, filename);
  return { blob, filename };
}
