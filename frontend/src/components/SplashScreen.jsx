import React, { useEffect, useState } from 'react';

export default function SplashScreen() {
  const [fadeOut, setFadeOut] = useState(false);
  const [hidden, setHidden] = useState(false);

  useEffect(() => {
    const timer1 = setTimeout(() => setFadeOut(true), 1000);
    const timer2 = setTimeout(() => setHidden(true), 1400);

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
    };
  }, []);

  if (hidden) return null;

  return (
    <div
      id="splashScreen"
      className={`splash-screen ${fadeOut ? 'fade-out' : ''}`}
    >
      <div className="splash-content">
        <div className="splash-icon-ring">
          <svg className="splash-ring-svg" viewBox="0 0 100 100">
            <circle
              cx="50"
              cy="50"
              r="45"
              fill="none"
              stroke="var(--border)"
              strokeWidth="2"
            />
            <circle
              cx="50"
              cy="50"
              r="45"
              fill="none"
              stroke="var(--accent)"
              strokeWidth="2"
              strokeDasharray="283"
              strokeDashoffset="283"
              className="splash-ring-progress"
            />
          </svg>
          <span className="splash-emoji">📈</span>
        </div>
        <div className="splash-logo">StockLSTM</div>
        <div className="splash-tagline">AI-Powered Forecasting</div>
      </div>
    </div>
  );
}
