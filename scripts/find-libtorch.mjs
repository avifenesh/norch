import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { execFileSync } from 'node:child_process';

function isValidLibTorchDir(dir) {
  if (!dir) return false;
  const libDir = path.join(dir, 'lib');
  const incDir = path.join(dir, 'include', 'ATen');
  if (!fs.existsSync(libDir) || !fs.existsSync(incDir)) return false;
  return (
    fs.existsSync(path.join(libDir, 'libtorch.so')) ||
    fs.existsSync(path.join(libDir, 'libtorch.dylib')) ||
    fs.existsSync(path.join(libDir, 'torch.dll'))
  );
}

function tryPythonProbe(pythonBin) {
  try {
    const out = execFileSync(
      pythonBin,
      ['-c', 'import torch, pathlib; print(pathlib.Path(torch.__file__).parent)'],
      { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'], timeout: 20000 }
    ).trim();
    if (isValidLibTorchDir(out)) return out;
  } catch {
    // ignore
  }
  return null;
}

function findInVenv(venvDir) {
  const libDir = path.join(venvDir, 'lib');
  if (!fs.existsSync(libDir)) return null;
  for (const entry of fs.readdirSync(libDir)) {
    if (!entry.startsWith('python')) continue;
    const candidate = path.join(libDir, entry, 'site-packages', 'torch');
    if (isValidLibTorchDir(candidate)) return path.resolve(candidate);
  }
  return null;
}

export function findLibTorchDir() {
  const candidates = [
    process.env.LIBTORCH,
    process.env.TORCH_INSTALL_PREFIX,
    path.join(os.homedir(), 'libtorch'),
    '/usr/local/libtorch',
    '/opt/libtorch',
  ].filter(Boolean);

  for (const candidate of candidates) {
    if (isValidLibTorchDir(candidate)) return path.resolve(candidate);
  }

  // Fast filesystem check for known local venvs before spawning Python
  const fastVenv = findInVenv(path.join(os.homedir(), 'projects', 'torch', '.venv'));
  if (fastVenv) return fastVenv;

  // Probe local Python environments with PyTorch installed
  const pythonCandidates = [
    process.env.NORCH_PYTHON,
    path.join(os.homedir(), 'projects', 'torch', '.venv', 'bin', 'python'),
    'python3',
    'python',
  ].filter(Boolean);

  for (const py of pythonCandidates) {
    const found = tryPythonProbe(py);
    if (found) return path.resolve(found);
  }

  throw new Error(
    'Could not locate libtorch. Set LIBTORCH=/path/to/libtorch (or install PyTorch in a Python environment and set NORCH_PYTHON).'
  );
}

if (import.meta.url === `file://${process.argv[1]}`) {
  console.log(findLibTorchDir());
}
