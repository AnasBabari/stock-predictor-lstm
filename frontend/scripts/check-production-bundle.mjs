import { readdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';

const assets = join(process.cwd(), 'dist', 'assets');
const files = await readdir(assets);
const forbiddenNames = files.filter((name) =>
  /trainingWorker|browserTrainingClient|tfjs/i.test(name)
);

const forbiddenContents = [];
for (const name of files.filter((entry) => entry.endsWith('.js'))) {
  const content = await readFile(join(assets, name), 'utf8');
  if (
    content.includes('TensorFlow.js')
    || content.includes('indexeddb://stocklstm')
    || content.includes('tfjs-return-lstm')
  ) {
    forbiddenContents.push(name);
  }
}

if (forbiddenNames.length || forbiddenContents.length) {
  throw new Error(
    `production bundle contains legacy browser training code; names=${forbiddenNames.join(',') || 'none'} contents=${forbiddenContents.join(',') || 'none'}`,
  );
}

console.log(`Production bundle is TFJS-free (${files.length} asset files inspected).`);
