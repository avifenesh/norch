import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..');
const buildDir = path.join(rootDir, 'build');
const srcDir = path.join(rootDir, 'native', 'src');

// Prepare a clang-tidy friendly compile_commands.json without GCC-specific PCH flags
const cdbDir = path.join(os.homedir(), '.cache', 'norch', 'clang-tidy-cdb');
fs.mkdirSync(cdbDir, { recursive: true });
const rawCdb = JSON.parse(fs.readFileSync(path.join(buildDir, 'compile_commands.json'), 'utf8'));
const cleanCdb = rawCdb.map((entry) => ({
  ...entry,
  command: entry.command
    .replace(/-Winvalid-pch\s*/g, '')
    .replace(/-include\s+\S+cmake_pch\.hxx\s*/g, ''),
}));
fs.writeFileSync(path.join(cdbDir, 'compile_commands.json'), JSON.stringify(cleanCdb, null, 2));

const files = fs
  .readdirSync(srcDir)
  .filter((f) => f.endsWith('.cpp'))
  .sort()
  .map((f) => path.join(srcDir, f));

const concurrency = 4;
let nextIndex = 0;
let failed = false;

function runOne(file) {
  return new Promise((resolve) => {
    const child = spawn('clang-tidy', ['-p', cdbDir, '--quiet', file], {
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let out = '';
    let err = '';
    child.stdout.on('data', (d) => (out += d.toString()));
    child.stderr.on('data', (d) => (err += d.toString()));
    child.on('close', (code) => {
      const name = path.basename(file);
      if (out.trim()) {
        process.stdout.write(`=== ${name} ===\n${out.trim()}\n`);
      }
      if (code !== 0) {
        failed = true;
        if (err.trim()) process.stderr.write(`=== ${name} stderr ===\n${err.trim()}\n`);
      }
      resolve();
    });
  });
}

async function worker() {
  while (nextIndex < files.length) {
    const idx = nextIndex++;
    await runOne(files[idx]);
  }
}

console.log(`[norch] running clang-tidy on ${files.length} files (concurrency=${concurrency})...`);
await Promise.all(Array.from({ length: Math.min(concurrency, files.length) }, () => worker()));
if (failed) {
  process.exit(1);
}
console.log('[norch] clang-tidy clean.');
