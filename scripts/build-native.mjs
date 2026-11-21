import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { findLibTorchDir } from './find-libtorch.mjs';

const require = createRequire(import.meta.url);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..');
const buildDir = path.join(rootDir, 'build');

const libtorchDir = findLibTorchDir();
const nodeIncludeDir = path.join(path.dirname(path.dirname(process.execPath)), 'include', 'node');
const napiDir = require('node-addon-api').include_dir;

if (!fs.existsSync(path.join(nodeIncludeDir, 'node_api.h'))) {
  throw new Error(`Node headers not found at ${nodeIncludeDir}`);
}

const hasCcache = (() => {
  try {
    execFileSync('which', ['ccache'], { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
})();

const hasNinja = (() => {
  try {
    execFileSync('which', ['ninja'], { stdio: 'ignore' });
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
  buildDir,
  ...(hasNinja ? ['-G', 'Ninja'] : []),
  `-DLIBTORCH_DIR=${libtorchDir}`,
  `-DNODE_INCLUDE_DIR=${nodeIncludeDir}`,
  `-DNODE_ADDON_API_DIR=${napiDir}`,
  '-DCMAKE_BUILD_TYPE=RelWithDebInfo',
  '-DCMAKE_CXX_SCAN_FOR_MODULES=OFF',
  '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
];

if (hasCcache) {
  cmakeArgs.push('-DCMAKE_CXX_COMPILER_LAUNCHER=ccache');
}

if (process.env.NORCH_SANITIZERS === '1') {
  cmakeArgs.push('-DNORCH_SANITIZERS=ON');
}

console.log(`[norch] libtorch: ${libtorchDir}`);
console.log(`[norch] node headers: ${nodeIncludeDir}`);
console.log(`[norch] configuring CMake (${hasNinja ? 'Ninja' : 'Make'}, jobs=${jobs})...`);

execFileSync('cmake', cmakeArgs, { stdio: 'inherit' });
execFileSync('cmake', ['--build', buildDir, '--parallel', String(jobs)], { stdio: 'inherit' });

const compileCommandsSrc = path.join(buildDir, 'compile_commands.json');
const compileCommandsDst = path.join(rootDir, 'compile_commands.json');
if (fs.existsSync(compileCommandsSrc)) {
  fs.copyFileSync(compileCommandsSrc, compileCommandsDst);
}

console.log(`[norch] built ${path.join(buildDir, 'norch_native.node')}`);
