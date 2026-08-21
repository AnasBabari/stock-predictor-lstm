import React from 'react';

export default function HeroSection() {
  return (
    <section className="hero">
      <div className="hero-badge">
        <span className="badge-dot"></span>
        Neural Network Forecasting
      </div>
      <h1>
        Stock Price <span className="gradient-text">Predictor</span>
      </h1>
      <p className="subtitle">
        Train a compact LSTM in your own browser on stationary features and compare it against
        persistence baselines on an untouched holdout — up to 30 trading days ahead. Educational
        project; forecasts are not financial advice.
      </p>
    </section>
  );
}
