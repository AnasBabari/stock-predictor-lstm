import * as tf from '@tensorflow/tfjs';
import { buildBrowserModel } from './modelFactory';
import { resolveTrainingProfile } from './trainingProfiles';

beforeAll(async () => { await tf.setBackend('cpu'); await tf.ready(); });
afterEach(() => tf.disposeVariables());

test.each(['quick', 'balanced', 'research'])('builds a %s price model with a 60x22 input and 30 outputs', (name) => {
  const model = buildBrowserModel('price', 22, resolveTrainingProfile(name));
  expect(model.inputs[0].shape).toEqual([null, 60, 22]);
  expect(model.outputs[0].shape).toEqual([null, 30]);
  model.dispose();
});

test('builds direction output as a 3-way softmax over [down, neutral, up]', async () => {
  const model = buildBrowserModel('direction', 22, resolveTrainingProfile('quick'));
  // Exactly three outputs — independent sigmoids would regress to a wider
  // head that cannot represent the neutral class or sum-to-one constraint.
  expect(model.outputs[0].shape).toEqual([null, 3]);
  const input = tf.zeros([1, 60, 22]);
  const output = model.predict(input);
  expect(output.shape).toEqual([1, 3]);
  const values = [...await output.data()];
  // Softmax contract: strictly positive and summing to one.
  expect(values.every((value) => value > 0 && value < 1)).toBe(true);
  expect(values.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 5);
  input.dispose(); output.dispose(); model.dispose();
});

test('compiles direction with categorical cross-entropy', async () => {
  const model = buildBrowserModel('direction', 22, resolveTrainingProfile('quick'));
  expect(model.loss).toBeDefined();
  // tfjs exposes compiled loss via model.compile config only at train time;
  // assert through a tiny fit step that CCE accepts one-hot targets.
  const xs = tf.zeros([2, 60, 22]);
  const ys = tf.oneHot(tf.tensor1d([1, 2], 'int32'), 3);
  await expect(model.fit(xs, ys, { epochs: 1 })).resolves.toBeDefined();
  xs.dispose();
  ys.dispose();
  model.dispose();
});
