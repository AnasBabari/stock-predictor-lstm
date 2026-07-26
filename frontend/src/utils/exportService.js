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
  if (
    priceData.predicted_prices?.length !== expectedDays ||
    directionData.directions?.length !== expectedDays ||
    directionData.probabilities?.length !== expectedDays
  ) {
    throw new Error('Forecast payload lengths do not match the requested export.');
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

  const rows = [['Date', 'Direction', 'Probability', 'Type']];
  stockData.future_dates.forEach((dt, i) => {
    rows.push([
      dt,
      stockData.directions?.[i] || '—',
      stockData.probabilities?.[i] != null ? stockData.probabilities[i].toFixed(4) : '—',
      'Predicted',
    ]);
  });

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

  const directionRows = [['Date', 'Direction', 'Probability', 'Type']];
  directionData.future_dates.forEach((dt, i) => {
    directionRows.push([
      dt,
      directionData.directions?.[i] || '—',
      directionData.probabilities?.[i] != null ? directionData.probabilities[i].toFixed(4) : '—',
      'Predicted',
    ]);
  });

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
