const API_BASE = import.meta.env.VITE_API_URL || window.STOCKLSTM_API_BASE || '';

/**
 * Fetches a server-pretrained forecast bundle.
 * Returns the parsed prediction object if available, or null if it falls back to browser training.
 */
export async function fetchServerPrediction(symbol, days, type, signal) {
  try {
    const response = await fetch(
      `${API_BASE}/api/v1/server-forecasts/${encodeURIComponent(symbol)}?forecast_type=${type}&days=${days}`,
      { signal, cache: 'no-cache' }
    );
    
    if (!response.ok) {
      return null; // Let the browser fallback take over for HTTP errors
    }
    
    const data = await response.json();
    
    // Check if the server explicitly directed us to fallback
    if (data.available === false && data.fallback === 'browser_training') {
      console.log(`Server prediction unavailable for ${symbol}: ${data.reason}. Falling back to browser training.`);
      return null;
    }
    
    // Format the bundle to match the expected standard output
    const daysInt = parseInt(days, 10);
    return {
      prices: data.predicted_prices.slice(0, daysInt),
      dates: data.future_dates.slice(0, daysInt),
      metrics: data.evidence || {},
      metadata: {
        server_pretrained: true,
        version_id: data.version_id,
        model_name: data.model_name || 'server_baseline',
        created_at: data.generated_at,
        browser_training: false,
        engine: {
          role: 'server_champion',
          baseline_fallback: false
        }
      }
    };
  } catch (error) {
    if (error.name === 'AbortError') {
      throw error; // Let AbortError propagate
    }
    console.error(`Failed to fetch server prediction for ${symbol}:`, error);
    return null; // Fallback on network errors
  }
}
