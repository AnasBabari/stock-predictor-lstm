import JSZip from 'jszip';

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.download = filename;
  link.href = url;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 500);
}

function csvFromRows(rows) {
  return rows.map((row) => row.join(',')).join('\n');
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
  downloadBlob(blob, `${metadata.ticker}_complete_analysis.zip`);
}