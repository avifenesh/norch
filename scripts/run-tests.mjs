import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..');
const testDir = path.join(rootDir, 'dist', 'test');

const cudaMode = process.argv.includes('--cuda');
const env = { ...process.env };

let files;
if (cudaMode) {
  env.NORCH_TEST_CUDA = '1';
  files = [path.join(testDir, 'cuda.test.js')];
} else {
  env.CUDA_VISIBLE_DEVICES = '';
  env.OMP_NUM_THREADS = env.OMP_NUM_THREADS || '4';
  files = fs
    .readdirSync(testDir)
    .filter((f) => f.endsWith('.test.js'))
    .sort()
    .map((f) => path.join(testDir, f));
}

const args = ['--test', ...(cudaMode ? [] : ['--test-concurrency=4']), ...files];
execFileSync(process.execPath, args, { cwd: rootDir, env, stdio: 'inherit' });
