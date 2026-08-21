import { describe, expect, it } from 'vitest';
import {
  DIRECTION_TARGET_VERSION,
  buildCumulativeDirectionTargets,
  directionThreshold,
  modelKey,
  trailingSigma20,
} from './preprocessing';

function makePrices(rows, seedFn) {
  const prices = [100];
  for (let i = 1; i < rows; i += 1) {
    prices.push(Math.max(1, prices[i - 1] * Math.exp(seedFn(i))));
  }
  return prices;
}

describe('directionThreshold', () => {
  it('uses the regulatory floor when volatility is negligible', () => {
    expect(directionThreshold({ sigma20: 0, horizon: 7, price: 100 })).toBeCloseTo(0.0005, 12);
  });

  it('volatility term dominates when 0.10·σ·√h exceeds the floor', () => {
    // σ=0.02 daily, h=25 → 0.10*0.02*5 = 0.01
    expect(directionThreshold({ sigma20: 0.02, horizon: 25, price: 100 })).toBeCloseTo(0.01, 12);
  });

  it('scales with √horizon', () => {
    const h1 = directionThreshold({ sigma20: 0.02, horizon: 1, price: 100 });
    const h4 = directionThreshold({ sigma20: 0.02, horizon: 4, price: 100 });
    expect(h4).toBeCloseTo(h1 * 2, 10);
  });

  it('tick term dominates for low-priced tickers', () => {
    // tick $0.01 at price $0.50 → 0.02, far above any vol term here
    const tau = directionThreshold({ sigma20: 0.01, horizon: 1, price: 0.5, tickPrice: 0.01 });
    expect(tau).toBeCloseTo(0.02, 12);
  });

  it('ignores invalid tick inputs', () => {
    const withoutTick = directionThreshold({ sigma20: 0, horizon: 1, price: 100 });
    expect(directionThreshold({ sigma20: 0, horizon: 1, price: 100, tickPrice: null })).toBe(withoutTick);
    expect(directionThreshold({ sigma20: 0, horizon: 1, price: 100, tickPrice: -1 })).toBe(withoutTick);
  });
});

describe('trailingSigma20', () => {
  it('is causal: perturbing the origin close and beyond cannot change σ', () => {
    const prices = makePrices(120, () => 0.001);
    const before = trailingSigma20(prices, 80);
    const perturbed = [...prices];
    for (let i = 80; i < perturbed.length; i += 1) perturbed[i] *= 3;
    expect(trailingSigma20(perturbed, 80)).toBe(before);
  });

  it('returns 0 without enough history', () => {
    expect(trailingSigma20([100, 101], 2)).toBe(0);
  });
});

describe('buildCumulativeDirectionTargets', () => {
  const rows = 200;
  // Strong deterministic up-trend: every cumulative label must be 'up'.
  const upPrices = makePrices(rows, () => 0.004);

  it('labels a persistent uptrend as up at every origin', () => {
    const labels = buildCumulativeDirectionTargets(upPrices, 7);
    expect(labels.length).toBeGreaterThan(50);
    expect(labels.every((entry) => entry.label === 'up')).toBe(true);
  });

  it('mirrors a downtrend to down', () => {
    const downPrices = makePrices(rows, () => -0.004);
    const labels = buildCumulativeDirectionTargets(downPrices, 7);
    expect(labels.every((entry) => entry.label === 'down')).toBe(true);
  });

  it('produces neutral inside the volatility band (flat series with noise below τ)', () => {
    // Tiny alternating noise: |r| far below the 0.0005 floor? Use exactly-flat.
    const flat = Array.from({ length: rows }, () => 100);
    const labels = buildCumulativeDirectionTargets(flat, 7);
    expect(labels.length).toBeGreaterThan(0);
    expect(labels.every((entry) => entry.label === 'neutral')).toBe(true);
    expect(labels.every((entry) => entry.cumulativeReturn === 0)).toBe(true);
  });

  it('never peeks past the origin: post-origin corruption only changes that origin’s own future label', () => {
    const clean = makePrices(rows, (i) => (i % 11 ? 0.0005 : -0.0006));
    const horizon = 7;
    const baseline = buildCumulativeDirectionTargets(clean, horizon);

    // Corrupt everything AFTER origin k's window but BEFORE its target close:
    // only origins whose TARGET window includes corrupted rows may change.
    const k = 60;
    const corrupted = [...clean];
    for (let i = k + 3; i < Math.min(k + horizon, corrupted.length); i += 1) {
      corrupted[i] *= 1.5;
    }
    const after = buildCumulativeDirectionTargets(corrupted, horizon);

    // Origins whose entire [origin, origin+h] span ends strictly before the
    // first corrupted row are untouched (target close AND sigma window).
    for (const entry of baseline) {
      if (entry.originIndex + horizon < k + 3) {
        const match = after.find((e) => e.originIndex === entry.originIndex);
        expect(match).toEqual(entry);
      }
    }
  });

  it('records the horizon and causal sigma on each label', () => {
    const first = buildCumulativeDirectionTargets(upPrices, 5)[0];
    expect(first.horizon).toBe(5);
    expect(first.sigma20).toBeGreaterThan(0);
    expect(first.tau).toBeGreaterThan(0);
    expect(first.originIndex).toBeGreaterThanOrEqual(21);
  });
});

describe('modelKey direction-target gating', () => {
  const snapshot = {
    schema_version: 4,
    ticker: 'TEST',
    feature_names: ['a'],
    snapshot_id: 'snap',
  };

  it('appends the direction target version only for direction artifacts', () => {
    const dirKey = modelKey(snapshot, 'direction', 'balanced', 'any', 7);
    const priceKey = modelKey(snapshot, 'price', 'balanced', 'any', 7);
    expect(dirKey.endsWith(`/${DIRECTION_TARGET_VERSION}`)).toBe(true);
    expect(priceKey.includes(DIRECTION_TARGET_VERSION)).toBe(false);
  });

  it('separates legacy per-day-sign direction caches from v2', () => {
    const legacyLike = modelKey(snapshot, 'direction', 'balanced', 'any', 7);
    expect(legacyLike).not.toContain('per_day_sign_v1');
    expect(new Set([legacyLike, legacyLike.replace(DIRECTION_TARGET_VERSION, 'per_day_sign_v1')]).size).toBe(2);
  });
});
