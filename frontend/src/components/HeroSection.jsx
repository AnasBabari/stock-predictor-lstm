import React from 'react';

export default function HeroSection() {
  return (
    <section className="hero">
      <div className="hero-badge">
        <span className="badge-dot"></span>
        Certified Volatility Forecasting
      </div>
      <h1>
        Stock Price <span className="gradient-text">Predictor</span>
      </h1>
      <p className="subtitle">
        A signed global model learns conditional volatility offline and serves a dated uncertainty
        cone around an unchanged-close location baseline — up to 30 trading sessions ahead.
        Educational project; forecasts are not financial advice.
      </p>
    </section>
  );
}
