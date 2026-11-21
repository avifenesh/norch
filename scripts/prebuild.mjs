import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..');
const builtAddon = path.join(rootDir, 'build', 'norch_native.node');
const pkg = JSON.parse(fs.readFileSync(path.join(rootDir, 'package.json'), 'utf8'));

if (!fs.existsSync(builtAddon)) {
  throw new Error(`Native addon not found at ${builtAddon}. Run "npm run build:native" first.`);
}

const triple = `${process.platform}-${process.arch}`;
const outDir = path.join(rootDir, 'prebuilds', triple);
fs.mkdirSync(outDir, { recursive: true });

const dstAddon = path.join(outDir, 'norch_native.node');
fs.copyFileSync(builtAddon, dstAddon);

const addon = require(dstAddon);
const sha256 = crypto.createHash('sha256').update(fs.readFileSync(dstAddon)).digest('hex');

const manifest = {
  name: pkg.name,
  version: pkg.version,
  platform: process.platform,
  arch: process.arch,
  napiVersion: 9,
  nodeVersion: process.version,
  libtorchVersion: addon.libtorchVersion(),
  cudaAvailableAtBuildCheck: Boolean(addon.NativeCudaStream),
  ncclCompiled: addon.distIsNcclAvailable !== undefined,
  sha256,
  builtAt: new Date().toISOString(),
};

fs.writeFileSync(path.join(outDir, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
console.log(`[norch] staged prebuild at ${dstAddon}`);
console.log(JSON.stringify(manifest, null, 2));

if (process.argv.includes('--pack')) {
  const tarName = `norch-v${pkg.version}-napi-v9-${triple}.tar.gz`;
  const tarPath = path.join(rootDir, 'prebuilds', tarName);
  execFileSync('tar', ['-czf', tarPath, '-C', path.join(rootDir, 'prebuilds'), triple], {
    stdio: 'inherit',
  });
  console.log(`[norch] packed ${tarPath}`);
}
