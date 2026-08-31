import React from 'react';

export default function HeroSection() {
  return (
    <section className="hero">
      <div className="hero-badge">
        <span className="badge-dot"></span>
        Causal Volatility Forecasting
      </div>
      <h1>
        Stock Price <span className="gradient-text">Predictor</span>
      </h1>
      <p className="subtitle">
        Transparent causal baselines estimate future realised volatility and serve a dated uncertainty
        cone around the latest close — up to 30 trading sessions ahead. Learned models are benchmarked
        offline before they can replace these baselines.
        Educational project; forecasts are not financial advice.
      </p>
    </section>
  );
}
