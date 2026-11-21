import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { findLibTorchDir } from './find-libtorch.mjs';

const require = createRequire(import.meta.url);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..');
const asanBuildDir = path.join(os.homedir(), '.cache', 'norch', 'build-asan');

const libtorchDir = findLibTorchDir();
const nodeIncludeDir = path.join(path.dirname(path.dirname(process.execPath)), 'include', 'node');
const napiDir = require('node-addon-api').include_dir;

const asanLib = execFileSync('g++', ['-print-file-name=libasan.so'], { encoding: 'utf8' }).trim();
if (!asanLib || !fs.existsSync(asanLib)) {
  throw new Error(`Could not locate libasan.so via g++ -print-file-name=libasan.so (got '${asanLib}')`);
}

const hasNinja = (() => {
  try {
    execFileSync('which', ['ninja'], { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
})();

const hasCcache = (() => {
  try {
    execFileSync('which', ['ccache'], { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
})();

const jobs = Math.max(1, Math.min(8, Math.floor(os.cpus().length / 2)));

const cmakeArgs = [
  '-S',
  rootDir,
  '-B',
  asanBuildDir,
  ...(hasNinja ? ['-G', 'Ninja'] : []),
  `-DLIBTORCH_DIR=${libtorchDir}`,
  `-DNODE_INCLUDE_DIR=${nodeIncludeDir}`,
  `-DNODE_ADDON_API_DIR=${napiDir}`,
  '-DCMAKE_BUILD_TYPE=RelWithDebInfo',
  '-DCMAKE_CXX_SCAN_FOR_MODULES=OFF',
  '-DNORCH_SANITIZERS=ON',
];
if (hasCcache) {
  cmakeArgs.push('-DCMAKE_CXX_COMPILER_LAUNCHER=ccache');
}

console.log(`[norch:asan] building instrumented addon into ${asanBuildDir}...`);
execFileSync('cmake', cmakeArgs, { stdio: 'inherit' });
execFileSync('cmake', ['--build', asanBuildDir, '--parallel', String(jobs)], { stdio: 'inherit' });

const asanAddon = path.join(asanBuildDir, 'norch_native.node');
const testFiles = [
  'dist/test/tensor_core.test.js',
  'dist/test/autograd.test.js',
  'dist/test/dispatcher.test.js',
  'dist/test/nn_optim.test.js',
  'dist/test/training_and_io.test.js',
  'dist/test/advanced_features.test.js',
  'dist/test/distributed_and_extras.test.js',
].map((rel) => path.join(rootDir, rel));

console.log(`[norch:asan] running native test suites under ASan+UBSan (${asanLib})...`);
const env = {
  ...process.env,
  CUDA_VISIBLE_DEVICES: '',
  LD_PRELOAD: process.env.LD_PRELOAD ? `${asanLib}:${process.env.LD_PRELOAD}` : asanLib,
  ASAN_OPTIONS: 'detect_leaks=0:halt_on_error=1:abort_on_error=1:handle_segv=0',
  UBSAN_OPTIONS: 'print_stacktrace=1:halt_on_error=1',
  NORCH_NATIVE_PATH: asanAddon,
  OMP_NUM_THREADS: '4',
};

execFileSync(process.execPath, ['--test', '--test-concurrency=2', ...testFiles], {
  cwd: rootDir,
  env,
  stdio: 'inherit',
});
console.log('[norch:asan] all ASan/UBSan test suites passed.');
