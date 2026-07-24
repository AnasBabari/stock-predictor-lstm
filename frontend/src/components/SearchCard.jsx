import React, { useState, useEffect, useRef, useCallback } from 'react';
import ForecastTypeToggle from './ForecastTypeToggle';

export default function SearchCard({
  ticker,
  setTicker,
  forecastDays,
  setForecastDays,
  forecastType,
  onForecastTypeChange,
  onPredict,
  isLoading,
  apiBase,
}) {
  const [suggestions, setSuggestions] = useState([]);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [highlightedIndex, setHighlightedIndex] = useState(-1);
  const searchWrapperRef = useRef(null);
  const searchTimeoutRef = useRef(null);

  const fetchSuggestions = useCallback(
    async (query) => {
      try {
        const res = await fetch(`${apiBase}/api/v1/search?query=${encodeURIComponent(query)}`);
        if (!res.ok) return;
        const data = await res.json();
        setSuggestions(data.results || []);
        setDropdownOpen(true);
        setHighlightedIndex(-1);
      } catch {
        // Ignore network errors on search autocomplete
      }
    },
    [apiBase]
  );

  const handleInputChange = (e) => {
    const value = e.target.value;
    setTicker(value);

    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current);
    }

    const trimmed = value.trim();
    if (trimmed.length < 2) {
      setSuggestions([]);
      setDropdownOpen(false);
      setHighlightedIndex(-1);
      return;
    }

    searchTimeoutRef.current = setTimeout(() => {
      fetchSuggestions(trimmed);
    }, 250);
  };

  const handleKeyDown = (e) => {
    if (!dropdownOpen || suggestions.length === 0) {
      if (e.key === 'Enter') {
        e.preventDefault();
        onPredict(ticker, forecastType);
      }
      return;
    }

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setHighlightedIndex((prev) => Math.min(suggestions.length - 1, prev + 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setHighlightedIndex((prev) => Math.max(-1, prev - 1));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (highlightedIndex >= 0 && highlightedIndex < suggestions.length) {
        const item = suggestions[highlightedIndex];
        setTicker(item.ticker);
        setDropdownOpen(false);
        setSuggestions([]);
        onPredict(item.ticker, forecastType);
      } else {
        onPredict(ticker, forecastType);
      }
    } else if (e.key === 'Escape') {
      setDropdownOpen(false);
      setHighlightedIndex(-1);
    }
  };

  const handleSuggestionClick = (item) => {
    setTicker(item.ticker);
    setDropdownOpen(false);
    setSuggestions([]);
    onPredict(item.ticker, forecastType);
  };

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (searchWrapperRef.current && !searchWrapperRef.current.contains(e.target)) {
        setDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      if (searchTimeoutRef.current) {
        clearTimeout(searchTimeoutRef.current);
      }
    };
  }, []);

  return (
    <section className="input-card glow-border" id="inputCard">
      <ForecastTypeToggle value={forecastType} onChange={onForecastTypeChange} />
      <div className="input-row">
        <div className="input-wrapper" ref={searchWrapperRef}>
          <label htmlFor="tickerInput" className="sr-only">
            Search stock ticker or company name
          </label>
          <svg
            className="input-svg-icon"
            viewBox="0 0 20 20"
            fill="currentColor"
            width="18"
            height="18"
          >
            <path
              fillRule="evenodd"
              d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z"
            />
          </svg>
          <input
            type="text"
            id="tickerInput"
            value={ticker}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder="Search tickers or company names…"
            maxLength={50}
            autoComplete="off"
            role="combobox"
            aria-expanded={dropdownOpen}
            aria-controls="searchDropdown"
            aria-activedescendant={
              highlightedIndex >= 0 ? `dropdown-item-${highlightedIndex}` : ''
            }
          />

          {dropdownOpen && suggestions.length > 0 && (
            <div id="searchDropdown" className="search-dropdown" role="listbox">
              {suggestions.map((r, i) => (
                <div
                  key={r.ticker}
                  id={`dropdown-item-${i}`}
                  className={`dropdown-item ${i === highlightedIndex ? 'highlighted' : ''}`}
                  style={{
                    background: i === highlightedIndex ? 'var(--accent-glow-subtle)' : undefined,
                  }}
                  role="option"
                  aria-selected={i === highlightedIndex}
                  onClick={() => handleSuggestionClick(r)}
                >
                  <div>
                    <div className="dropdown-name">{r.name}</div>
                    <div className="dropdown-type">{r.type}</div>
                  </div>
                  <div className="dropdown-ticker">{r.ticker}</div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="forecast-selector">
          <label htmlFor="forecastDays" className="sr-only">
            Forecast horizon
          </label>
          <select
            id="forecastDays"
            title="Forecast horizon"
            value={forecastDays}
            onChange={(e) => setForecastDays(Number(e.target.value))}
          >
            <option value={3}>3 days</option>
            <option value={7}>7 days</option>
            <option value={14}>14 days</option>
            <option value={30}>30 days</option>
          </select>
        </div>

        <button
          type="button"
          id="predictBtn"
          className="predict-btn"
          onClick={() => onPredict(ticker, forecastType)}
          disabled={isLoading}
        >
          <span className="btn-text">Predict</span>
          <span className="btn-icon">
            <svg viewBox="0 0 20 20" fill="currentColor" width="16" height="16">
              <path
                fillRule="evenodd"
                d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z"
              />
            </svg>
          </span>
          <span className="btn-shine"></span>
        </button>
      </div>
    </section>
  );
}
