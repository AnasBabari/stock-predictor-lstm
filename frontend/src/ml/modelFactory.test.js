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

test('builds direction output with sigmoid probabilities', async () => {
  const model = buildBrowserModel('direction', 22, resolveTrainingProfile('quick'));
  const input = tf.zeros([1, 60, 22]);
  const output = model.predict(input);
  const values = await output.data();
  expect([...values].every((value) => value >= 0 && value <= 1)).toBe(true);
  input.dispose(); output.dispose(); model.dispose();
});
