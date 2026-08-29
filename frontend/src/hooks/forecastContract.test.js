import { describe, expect, it } from 'vitest';
import { describePromotionState } from '../ml/promotionPolicy';

describe('describePromotionState v2', () => {
  it('maps a promoted model to promoted state with decision model and alpha 1', () => {
    const status = describePromotionState({ promoted: true, applicable: true });
    expect(status).toMatchObject({ state: 'promoted', decision: 'model', alpha: 1 });
  });

  it('maps rejection to experimental state with decision model and alpha 1 (model line preserved)', () => {
    const status = describePromotionState({ promoted: false, state: 'experimental', applicable: true });
    expect(status.state).toBe('experimental');
    expect(status.decision).toBe('model');
    expect(status.alpha).toBe(1);
    expect(status.label).toMatch(/validation gates were not met/i);
  });

  it('fails closed for missing or non-applicable gates to unavailable state with decision model', () => {
    const unknown = describePromotionState(null);
    expect(unknown.state).toBe('unavailable');
    expect(unknown.decision).toBe('model');
    expect(unknown.alpha).toBe(1);

    const notApplicable = describePromotionState({ promoted: true, applicable: false });
    expect(notApplicable.decision).toBe('model');
    expect(notApplicable.state).toBe('unavailable');
  });
});
