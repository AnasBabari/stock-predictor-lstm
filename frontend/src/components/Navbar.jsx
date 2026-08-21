import React from 'react';

export default function Navbar({ theme, onToggleTheme, onClearBrowserModels }) {
  return (
    <header className="navbar" id="navbar">
      <div className="nav-brand">
        <span className="nav-logo-icon">📈</span>
        <span className="nav-logo-text">
          Stock<span className="nav-logo-accent">LSTM</span>
        </span>
      </div>
      <div className="nav-actions">
        {onClearBrowserModels && (
          <button
            type="button"
            className="icon-btn clear-models-btn"
            onClick={onClearBrowserModels}
            aria-label="Clear locally trained browser models"
            title="Clear locally trained browser models"
          >
            <span aria-hidden="true">🧹</span>
          </button>
        )}
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
