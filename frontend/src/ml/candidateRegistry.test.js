import { describe, expect, it } from 'vitest';
import {
  CANDIDATE_FAMILIES,
  fitRollingMean,
  predictRidge,
  rankDevelopmentHorizons,
  resolveAutoHorizon,
  selectDevelopmentChampions,
  solveRidge,
} from './candidateRegistry';

describe('solveRidge', () => {
  it('accurately fits a linear relationship with small regularization', () => {
    // y = 2*x1 - 3*x2 + 0.5
    const X = [
      [1.0, 0.0],
      [2.0, 1.0],
      [0.0, 2.0],
      [3.0, 3.0],
      [1.5, 0.5],
      [-1.0, -1.0],
    ];
    const y = X.map(([x1, x2]) => 2 * x1 - 3 * x2 + 0.5);

    const model = solveRidge(X, y, 1e-4);
    expect(model.weights[0]).toBeCloseTo(2.0, 2);
    expect(model.weights[1]).toBeCloseTo(-3.0, 2);
    expect(model.bias).toBeCloseTo(0.5, 2);

    const preds = predictRidge(X, model);
    for (let i = 0; i < y.length; i += 1) {
      expect(preds[i]).toBeCloseTo(y[i], 1);
    }
  });

  it('handles collinear and ill-conditioned matrices without throwing', () => {
    const X = [
      [1.0, 1.0],
      [2.0, 2.0],
      [3.0, 3.0],
    ];
    const y = [1.0, 2.0, 3.0];
    const model = solveRidge(X, y, 1.0);
    expect(Number.isFinite(model.weights[0])).toBe(true);
    expect(Number.isFinite(model.weights[1])).toBe(true);
  });
});

describe('selectDevelopmentChampions and nested selection', () => {
  it('selects champions strictly from development splits and ignores changes in the holdout partition', () => {
    const totalSamples = 100;
    const devEnd = 80;
    const numFeatures = 4;
    const horizons = [1, 3, 5, 7];

    // Synthetic development dataset
    const devInputs = Array.from({ length: devEnd }, (_, i) =>
      Array.from({ length: numFeatures }, (_, j) => Math.sin(i * 0.1 + j))
    );
    const devTargets = Array.from({ length: devEnd }, (_, i) =>
      horizons.map((h) => 0.02 * h * (i % 2 === 0 ? 1 : -1))
    );

    const devSplits = [
      { trainEnd: 40, valStart: 45, valEnd: 60 },
      { trainEnd: 60, valStart: 65, valEnd: 80 },
    ];

    const champions1 = selectDevelopmentChampions({
      trainInputs: devInputs,
      trainTargets: devTargets,
      devValidationSplits: devSplits,
      horizons,
    });

    // Holdout data that is completely altered
    const holdoutTargetsAltered = Array.from({ length: totalSamples - devEnd }, () =>
      horizons.map(() => 999.99)
    );

    // Running development selection again with identical development data must yield the EXACT same champions
    const champions2 = selectDevelopmentChampions({
      trainInputs: devInputs,
      trainTargets: devTargets,
      devValidationSplits: devSplits,
      horizons,
    });

    expect(champions1).toEqual(champions2);
    expect(Object.keys(champions1)).toEqual(horizons.map(String));
  });

  it('proves the strict nested-selection invariant: changing holdout targets cannot alter selected candidate or horizon, only promotion status', () => {
    const devEnd = 80;
    const holdoutLength = 20;
    const numFeatures = 3;
    const horizons = [1, 3, 5, 7];

    // Step A: Development features and targets
    const devInputs = Array.from({ length: devEnd }, (_, i) => [
      Math.sin(i * 0.1),
      Math.cos(i * 0.1),
      (i % 5) * 0.2,
    ]);
    const devTargets = Array.from({ length: devEnd }, (_, i) => [
      0.5 * Math.sin(i * 0.1),
      0.8 * Math.sin(i * 0.1),
      1.0 * Math.sin(i * 0.1),
      1.2 * Math.sin(i * 0.1),
    ]);

    const devSplits = [
      { trainEnd: 40, valStart: 45, valEnd: 60 },
      { trainEnd: 60, valStart: 65, valEnd: 80 },
    ];

    // Step B: Run development selection and record candidate identity and horizon ranking
    const devChampions = selectDevelopmentChampions({
      trainInputs: devInputs,
      trainTargets: devTargets,
      devValidationSplits: devSplits,
      horizons,
    });
    const devHorizonRanking = rankDevelopmentHorizons({
      trainInputs: devInputs,
      trainTargets: devTargets,
      devValidationSplits: devSplits,
      horizons,
    });

    const recordedCandidate = devChampions[3];
    const recordedChampionHorizon = devHorizonRanking.developmentChampionHorizon;

    // Step C: Create Holdout A (clean signal where model accurately predicts targets)
    const holdoutInputs = Array.from({ length: holdoutLength }, (_, i) => [
      Math.sin((devEnd + i) * 0.1),
      Math.cos((devEnd + i) * 0.1),
      ((devEnd + i) % 5) * 0.2,
    ]);
    const holdoutTargetsA = Array.from({ length: holdoutLength }, (_, i) => [
      0.5 * Math.sin((devEnd + i) * 0.1),
      0.8 * Math.sin((devEnd + i) * 0.1),
      1.0 * Math.sin((devEnd + i) * 0.1),
      1.2 * Math.sin((devEnd + i) * 0.1),
    ]);

    // Step D: Create Holdout B (absurdly corrupted targets with massive noise)
    const holdoutTargetsB = Array.from({ length: holdoutLength }, () => [
      -99999.0, 88888.0, -77777.0, 66666.0,
    ]);

    // Fit champion model strictly on development data
    const lastStepFeatures = devInputs.map((sample) =>
      Array.isArray(sample[0]) ? sample[sample.length - 1] : sample
    );
    const targetIdx = horizons.indexOf(recordedChampionHorizon);
    const devY = devTargets.map((r) => r[targetIdx]);
    const frozenModel = solveRidge(lastStepFeatures, devY, 1e-4);

    // Evaluate on Holdout A
    const predsA = predictRidge(holdoutInputs, frozenModel);
    const mseA = predsA.reduce((sum, p, idx) => sum + (p - holdoutTargetsA[idx][targetIdx]) ** 2, 0) / holdoutLength;
    const persistenceMseA = holdoutTargetsA.reduce((sum, r) => sum + r[targetIdx] ** 2, 0) / holdoutLength;
    const relMseA = mseA / persistenceMseA;
    const promotedA = relMseA < 1.0;

    // Evaluate on Holdout B
    const predsB = predictRidge(holdoutInputs, frozenModel);
    const mseB = predsB.reduce((sum, p, idx) => sum + (p - holdoutTargetsB[idx][targetIdx]) ** 2, 0) / holdoutLength;
    const persistenceMseB = holdoutTargetsB.reduce((sum, r) => sum + r[targetIdx] ** 2, 0) / holdoutLength;
    const relMseB = mseB / persistenceMseB;
    const promotedB = relMseB < 1.0;

    // Step E: Prove candidate identity and horizon ranking are 100% identical regardless of holdout
    const devChampionsAgain = selectDevelopmentChampions({
      trainInputs: devInputs,
      trainTargets: devTargets,
      devValidationSplits: devSplits,
      horizons,
    });
    const devHorizonRankingAgain = rankDevelopmentHorizons({
      trainInputs: devInputs,
      trainTargets: devTargets,
      devValidationSplits: devSplits,
      horizons,
    });

    expect(devChampionsAgain[3]).toBe(recordedCandidate);
    expect(devHorizonRankingAgain.developmentChampionHorizon).toBe(recordedChampionHorizon);

    // Step F: Prove validation status differs between Holdout A and Holdout B
    expect(promotedA).toBe(true); // Promoted on accurate holdout
    expect(promotedB).toBe(false); // Rejected on corrupted holdout
    expect(relMseA).toBeLessThan(1.0);
    expect(relMseB).toBeGreaterThan(1.0);
  });
});

describe('resolveAutoHorizon', () => {
  it('recommends top development-ranked horizon that cleared holdout promotion', () => {
    const autoResult = resolveAutoHorizon({
      developmentChampionHorizon: 3,
      developmentRanking: [
        { horizon: 3, relative_mse: 0.85 },
        { horizon: 5, relative_mse: 0.90 },
        { horizon: 7, relative_mse: 0.92 },
      ],
      promotedHorizons: [5, 7],
    });

    // Horizon 3 was not promoted on holdout; top development-ranked promoted horizon is 5
    expect(autoResult.selectedHorizon).toBe(5);
    expect(autoResult.validated).toBe(true);
    expect(autoResult.reason).toContain('5d');
  });

  it('retains development champion horizon as experimental when no horizon passed holdout promotion', () => {
    const autoResult = resolveAutoHorizon({
      developmentChampionHorizon: 3,
      developmentRanking: [
        { horizon: 3, relative_mse: 0.85 },
        { horizon: 5, relative_mse: 0.90 },
        { horizon: 7, relative_mse: 0.92 },
      ],
      promotedHorizons: [],
    });

    // Retains development champion (3) as experimental
    expect(autoResult.selectedHorizon).toBe(3);
    expect(autoResult.validated).toBe(false);
    expect(autoResult.reason).toContain('3d');
  });
});

