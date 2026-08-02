import * as tf from '@tensorflow/tfjs';
import { OUTPUT_WIDTH, WINDOW_SIZE } from './preprocessing';

export function buildBrowserModel(forecastType, featureCount, profile) {
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
  model.add(tf.layers.dense({ units: profile.denseUnits, activation: 'relu' }));
  model.add(tf.layers.dense({
    units: OUTPUT_WIDTH,
    activation: forecastType === 'direction' ? 'sigmoid' : undefined,
  }));
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: forecastType === 'direction' ? 'binaryCrossentropy' : 'meanSquaredError',
  });
  return model;
}
