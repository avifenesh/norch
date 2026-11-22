#!/usr/bin/env node
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';

export const DEFAULT_LIBTORCH_VERSION = '2.7.0';

/**
 * Resolves the official PyTorch libtorch zip download URL for a given platform/backend.
 */
export function resolveLibTorchUrl({
  platform = process.platform,
  arch = process.arch,
  backend = 'cpu',
  version = DEFAULT_LIBTORCH_VERSION,
} = {}) {
  const normalizedBackend = backend.toLowerCase();
  if (platform === 'linux') {
    if (arch !== 'x64') {
      throw new Error(`Prebuilt Linux libtorch zips require x64 (got ${arch}); install via Python wheel or build from source.`);
    }
    const valid = new Set(['cpu', 'cu124', 'cu126', 'cu128', 'rocm6.2']);
    if (!valid.has(normalizedBackend)) {
      throw new Error(`Unsupported Linux backend '${backend}'. Expected one of: ${Array.from(valid).join(', ')}`);
    }
    const suffix = normalizedBackend === 'cpu' ? '%2Bcpu' : `%2B${normalizedBackend}`;
    const filename = `libtorch-cxx11-abi-shared-with-deps-${version}${suffix}.zip`;
    return {
      url: `https://download.pytorch.org/libtorch/${normalizedBackend}/${filename}`,
      filename,
      platform,
      arch,
      backend: normalizedBackend,
      version,
    };
  }

  if (platform === 'darwin') {
    const filename =
      arch === 'arm64'
        ? `libtorch-macos-arm64-${version}.zip`
        : `libtorch-macos-x86_64-${version}.zip`;
    return {
      url: `https://download.pytorch.org/libtorch/cpu/${filename}`,
      filename,
      platform,
      arch,
      backend: 'mps',
      version,
    };
  }

  if (platform === 'win32') {
    const valid = new Set(['cpu', 'cu124', 'cu126', 'cu128']);
    if (!valid.has(normalizedBackend)) {
      throw new Error(`Unsupported Windows backend '${backend}'. Expected one of: ${Array.from(valid).join(', ')}`);
    }
    const suffix = normalizedBackend === 'cpu' ? '%2Bcpu' : `%2B${normalizedBackend}`;
    const filename = `libtorch-win-shared-with-deps-${version}${suffix}.zip`;
    return {
      url: `https://download.pytorch.org/libtorch/${normalizedBackend}/${filename}`,
      filename,
      platform,
      arch,
      backend: normalizedBackend,
      version,
    };
  }

  throw new Error(`Unsupported platform: ${platform}`);
}

export function parseArgs(argv) {
  const opts = {
    backend: 'cpu',
    version: DEFAULT_LIBTORCH_VERSION,
    dest: path.join(os.homedir(), 'libtorch'),
    platform: process.platform,
    arch: process.arch,
    dryRun: false,
    json: false,
  };

  for (const arg of argv) {
    if (arg === '--dry-run') {
      opts.dryRun = true;
    } else if (arg === '--json') {
      opts.json = true;
    } else if (arg.startsWith('--backend=')) {
      opts.backend = arg.slice('--backend='.length);
    } else if (arg.startsWith('--version=')) {
      opts.version = arg.slice('--version='.length);
    } else if (arg.startsWith('--dest=')) {
      opts.dest = path.resolve(arg.slice('--dest='.length));
    } else if (arg.startsWith('--platform=')) {
      opts.platform = arg.slice('--platform='.length);
    } else if (arg.startsWith('--arch=')) {
      opts.arch = arg.slice('--arch='.length);
    } else if (arg === '--help' || arg === '-h') {
      console.log(
        [
          'Usage: node scripts/download-libtorch.mjs [options]',
          '  --backend=cpu|cu124|cu126|cu128|mps   Compute backend (default: cpu)',
          `  --version=X.Y.Z                       LibTorch version (default: ${DEFAULT_LIBTORCH_VERSION})`,
          '  --dest=/path/to/libtorch              Install destination (default: ~/libtorch)',
          '  --platform=linux|darwin|win32         Override target platform',
          '  --arch=x64|arm64                      Override target architecture',
          '  --dry-run                             Print resolved URL and destination without downloading',
          '  --json                                Output machine-readable JSON',
        ].join('\n')
      );
      process.exit(0);
    }
  }
  return opts;
}

export async function downloadLibTorch(opts) {
  const info = resolveLibTorchUrl(opts);
  const plan = {
    ...info,
    dest: opts.dest,
    dryRun: opts.dryRun,
  };

  if (opts.dryRun) {
    if (opts.json) {
      console.log(JSON.stringify(plan, null, 2));
    } else {
      console.log(`[norch] dry-run libtorch download:`);
      console.log(`  url:     ${plan.url}`);
      console.log(`  dest:    ${plan.dest}`);
      console.log(`  backend: ${plan.backend} (${plan.platform}/${plan.arch}, v${plan.version})`);
    }
    return plan;
  }

  const cacheDir = path.join(os.homedir(), '.cache', 'norch', 'downloads');
  fs.mkdirSync(cacheDir, { recursive: true });
  const zipPath = path.join(cacheDir, info.filename.replace(/%2B/g, '+'));

  console.log(`[norch] downloading ${info.url} -> ${zipPath}`);
  const res = await fetch(info.url);
  if (!res.ok || !res.body) {
    throw new Error(`Failed to download libtorch: HTTP ${res.status} ${res.statusText}`);
  }
  await pipeline(Readable.fromWeb(res.body), fs.createWriteStream(zipPath));

  const parentDir = path.dirname(opts.dest);
  fs.mkdirSync(parentDir, { recursive: true });
  console.log(`[norch] extracting into ${parentDir}...`);
  if (process.platform === 'win32') {
    execFileSync('tar', ['-xf', zipPath, '-C', parentDir], { stdio: 'inherit' });
  } else {
    execFileSync('unzip', ['-qo', zipPath, '-d', parentDir], { stdio: 'inherit' });
  }

  const extractedDefault = path.join(parentDir, 'libtorch');
  if (path.resolve(extractedDefault) !== path.resolve(opts.dest) && fs.existsSync(extractedDefault)) {
    fs.rmSync(opts.dest, { recursive: true, force: true });
    fs.renameSync(extractedDefault, opts.dest);
  }
  console.log(`[norch] libtorch ready at ${opts.dest}`);
  console.log(`[norch] export LIBTORCH="${opts.dest}" && npm run build`);
  return plan;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const opts = parseArgs(process.argv.slice(2));
  await downloadLibTorch(opts);
}
