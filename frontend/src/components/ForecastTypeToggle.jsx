import React from 'react';

export default function ForecastTypeToggle({ value, onChange }) {
  return (
    <section className="forecast-type-card">
      <div className="section-label">Forecast Type</div>
      <div className="forecast-type-toggle" role="radiogroup" aria-label="Forecast Type">
        <button
          type="button"
          className={`toggle-pill ${value === 'price' ? 'active' : ''}`}
          aria-pressed={value === 'price'}
          onClick={() => onChange('price')}
        >
          Price Forecast
        </button>
        <button
          type="button"
          className={`toggle-pill ${value === 'trend' ? 'active' : ''}`}
          aria-pressed={value === 'trend'}
          onClick={() => onChange('trend')}
        >
          Trend Forecast
        </button>
      </div>
    </section>
  );
}