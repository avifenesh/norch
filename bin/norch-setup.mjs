#!/usr/bin/env node
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { findLibTorchDir } from '../scripts/find-libtorch.mjs';
import { downloadLibTorch, parseArgs } from '../scripts/download-libtorch.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..');

const [command = 'help', ...rest] = process.argv.slice(2);

if (command === 'detect') {
  try {
    const dir = findLibTorchDir();
    console.log(dir);
  } catch (err) {
    console.error(err instanceof Error ? err.message : String(err));
    process.exit(1);
  }
} else if (command === 'download') {
  const opts = parseArgs(rest);
  await downloadLibTorch(opts);
} else if (command === 'build') {
  execFileSync(process.execPath, [path.join(rootDir, 'scripts', 'build-native.mjs')], {
    stdio: 'inherit',
  });
} else {
  console.log(
    [
      'norch-setup: manage libtorch and native C++23 addon builds',
      '',
      'Commands:',
      '  norch-setup detect                         Locate an installed libtorch directory',
      '  norch-setup download [--backend=cpu|cu128] [--version=2.7.0] [--dest=~/libtorch] [--dry-run]',
      '  norch-setup build                          Compile build/norch_native.node against libtorch',
    ].join('\n')
  );
}
