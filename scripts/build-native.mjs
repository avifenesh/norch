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

const isWindows = process.platform === 'win32';
const libtorchDir = findLibTorchDir();
const napiDir = require('node-addon-api').include_dir;

function hasTool(name) {
  try {
    execFileSync(isWindows ? 'where' : 'which', [name], { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

async function ensureNodeHeaders() {
  const candidates = [
    process.env.NORCH_NODE_INCLUDE_DIR,
    path.join(path.dirname(path.dirname(process.execPath)), 'include', 'node'),
    path.join(path.dirname(process.execPath), 'include', 'node'),
    path.join(os.homedir(), '.cache', 'norch', 'node-headers', process.version, 'include', 'node'),
  ].filter(Boolean);

  for (const dir of candidates) {
    if (fs.existsSync(path.join(dir, 'node_api.h'))) {
      return dir;
    }
  }

  // Fallback for Windows/minimal Node installs that omit bundled headers: fetch official headers
  const cacheRoot = path.join(os.homedir(), '.cache', 'norch', 'node-headers', process.version);
  fs.mkdirSync(cacheRoot, { recursive: true });
  const tarballUrl = `https://nodejs.org/dist/${process.version}/node-${process.version}-headers.tar.gz`;
  const tarPath = path.join(cacheRoot, 'headers.tar.gz');
  console.log(`[norch] downloading Node headers ${tarballUrl}...`);
  const res = await fetch(tarballUrl);
  if (!res.ok) {
    throw new Error(`Node headers not found locally and failed to download ${tarballUrl}: HTTP ${res.status}`);
  }
  const buf = Buffer.from(await res.arrayBuffer());
  fs.writeFileSync(tarPath, buf);
  execFileSync('tar', ['-xzf', tarPath, '-C', cacheRoot, '--strip-components=1'], { stdio: 'inherit' });
  const extracted = path.join(cacheRoot, 'include', 'node');
  if (!fs.existsSync(path.join(extracted, 'node_api.h'))) {
    throw new Error(`Node headers still missing at ${extracted}`);
  }
  return extracted;
}

async function ensureWindowsNodeLib() {
  if (!isWindows) return null;
  const cacheDir = path.join(os.homedir(), '.cache', 'norch', 'node-headers', process.version);
  fs.mkdirSync(cacheDir, { recursive: true });
  const nodeLibPath = path.join(cacheDir, 'node.lib');
  if (fs.existsSync(nodeLibPath)) return nodeLibPath;
  const archDir = process.arch === 'arm64' ? 'win-arm64' : 'win-x64';
  const url = `https://nodejs.org/dist/${process.version}/${archDir}/node.lib`;
  console.log(`[norch] downloading Windows import library ${url}...`);
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to download ${url}: HTTP ${res.status}`);
  }
  fs.writeFileSync(nodeLibPath, Buffer.from(await res.arrayBuffer()));
  return nodeLibPath;
}

const nodeIncludeDir = await ensureNodeHeaders();
const nodeLibPath = await ensureWindowsNodeLib();

const hasCcache = !isWindows && hasTool('ccache');
const hasNinja = !isWindows && hasTool('ninja');

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

if (nodeLibPath) {
  cmakeArgs.push(`-DNODE_LIB=${nodeLibPath}`);
}

if (hasCcache) {
  cmakeArgs.push('-DCMAKE_CXX_COMPILER_LAUNCHER=ccache');
}

if (process.env.NORCH_SANITIZERS === '1') {
  cmakeArgs.push('-DNORCH_SANITIZERS=ON');
}

console.log(`[norch] libtorch: ${libtorchDir}`);
console.log(`[norch] node headers: ${nodeIncludeDir}`);
console.log(`[norch] configuring CMake (${hasNinja ? 'Ninja' : isWindows ? 'Visual Studio' : 'Make'}, jobs=${jobs})...`);

execFileSync('cmake', cmakeArgs, { stdio: 'inherit' });
const buildArgs = ['--build', buildDir, '--parallel', String(jobs)];
if (isWindows) {
  buildArgs.push('--config', 'Release');
}
execFileSync('cmake', buildArgs, { stdio: 'inherit' });

const compileCommandsSrc = path.join(buildDir, 'compile_commands.json');
const compileCommandsDst = path.join(rootDir, 'compile_commands.json');
if (fs.existsSync(compileCommandsSrc)) {
  fs.copyFileSync(compileCommandsSrc, compileCommandsDst);
}

console.log(`[norch] built ${path.join(buildDir, 'norch_native.node')}`);
