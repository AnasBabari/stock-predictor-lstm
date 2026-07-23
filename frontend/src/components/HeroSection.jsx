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
        Harness the power of LSTM deep learning to forecast any stock ticker — up to 30 trading days ahead
      </p>
    </section>
  );
}
