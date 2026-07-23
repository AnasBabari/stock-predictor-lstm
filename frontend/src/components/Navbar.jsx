import React from 'react';

export default function Navbar({ theme, onToggleTheme }) {
  return (
    <header className="navbar" id="navbar">
      <div className="nav-brand">
        <span className="nav-logo-icon">📈</span>
        <span className="nav-logo-text">
          Stock<span className="nav-logo-accent">LSTM</span>
        </span>
      </div>
      <div className="nav-actions">
        <button
          type="button"
          className="icon-btn theme-btn"
          onClick={onToggleTheme}
          aria-label="Toggle theme"
          title="Toggle theme"
        >
          <span className="theme-icon">{theme === 'dark' ? '🌙' : '☀️'}</span>
        </button>
      </div>
    </header>
  );
}
