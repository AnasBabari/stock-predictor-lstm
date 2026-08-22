import * as tf from '@tensorflow/tfjs';
import { DIRECTION_CLASSES, WINDOW_SIZE } from './preprocessing';

export const HUBER_DELTA = 0.02;

// Direction v2 head: ONE three-way softmax over the cumulative horizon —
// class order [down, neutral, up] (DIRECTION_CLASSES), trained with
// categorical cross-entropy against integer class targets one-hot encoded
// by the worker.
export const DIRECTION_OUTPUT_UNITS = DIRECTION_CLASSES.length;

export function buildBrowserModel(forecastType, featureCount, profile, outputWidth = 30) {
  const isDirection = forecastType === 'direction' || forecastType === 'trend';
  const model = tf.sequential();
  model.add(tf.layers.lstm({
    units: profile.lstmUnits[0],
    returnSequences: true,
    inputShape: [WINDOW_SIZE, featureCount],
    kernelInitializer: tf.initializers.glorotUniform({ seed: 42 }),
    recurrentInitializer: tf.initializers.orthogonal({ seed: 43 }),
  }));
  model.add(tf.layers.dropout({ rate: profile.dropout, seed: 44 }));
  model.add(tf.layers.lstm({
    units: profile.lstmUnits[1],
    kernelInitializer: tf.initializers.glorotUniform({ seed: 45 }),
    recurrentInitializer: tf.initializers.orthogonal({ seed: 46 }),
  }));
  if (profile.id !== 'quick') model.add(tf.layers.dropout({ rate: profile.dropout, seed: 47 }));
  model.add(tf.layers.dense({
    units: profile.denseUnits,
    activation: 'relu',
    kernelInitializer: tf.initializers.glorotUniform({ seed: 48 }),
  }));
  model.add(tf.layers.dense({
    units: isDirection ? DIRECTION_OUTPUT_UNITS : Math.max(1, Math.round(Number(outputWidth) || 30)),
    activation: isDirection ? 'softmax' : undefined,
    kernelInitializer: tf.initializers.glorotUniform({ seed: 49 }),
  }));
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: isDirection ? 'categoricalCrossentropy' : (yTrue, yPred) => tf.losses.huberLoss(yTrue, yPred, HUBER_DELTA),
  });
  return model;
}
