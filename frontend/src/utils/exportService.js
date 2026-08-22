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
