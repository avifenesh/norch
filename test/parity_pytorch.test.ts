import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, it } from 'node:test';
import torch, { F, optim } from '../src/index.js';

function pythonSubprocessEnv(): NodeJS.ProcessEnv {
  const env = { ...process.env };
  delete env.LD_PRELOAD;
  delete env.ASAN_OPTIONS;
  delete env.UBSAN_OPTIONS;
  return env;
}

function findPythonTorch(): string | null {
  const candidates = [
    process.env.NORCH_PYTHON,
    path.join(os.homedir(), 'projects', 'torch', '.venv', 'bin', 'python'),
    'python3',
  ].filter((p): p is string => Boolean(p));

  for (const py of candidates) {
    if (py.includes(path.sep) && !fs.existsSync(py)) continue;
    try {
      execFileSync(py, ['-c', 'import torch'], {
        stdio: 'ignore',
        timeout: 5000,
        env: pythonSubprocessEnv(),
      });
      return py;
    } catch {
      // ignore
    }
  }
  return null;
}

function assertCloseArray(actual: number[], expected: number[], tol = 1e-5, label = ''): void {
  assert.equal(actual.length, expected.length, `${label} length mismatch`);
  for (let i = 0; i < actual.length; i++) {
    const diff = Math.abs(actual[i]! - expected[i]!);
    assert.ok(
      diff <= tol,
      `${label}[${i}] mismatch: norch=${actual[i]} vs pytorch=${expected[i]} (diff=${diff})`
    );
  }
}

describe('Numerical parity against Python PyTorch', () => {
  const py = findPythonTorch();

  it('matches Python PyTorch forward & backward for linear, conv2d, layerNorm, rmsNorm, SDPA, crossEntropy, and AdamW', (t) => {
    if (!py) {
      t.skip('No local Python PyTorch installation found for cross-validation');
      return;
    }

    const pyScript = `
import json, torch, torch.nn.functional as F

# 1. Linear + LayerNorm + CrossEntropy
x = torch.tensor([[0.5, -1.2, 0.3, 2.1], [-0.4, 0.8, 1.5, -0.7]], dtype=torch.float32, requires_grad=True)
w = torch.tensor([
    [0.2, -0.3, 0.4, 0.1],
    [-0.5, 0.6, -0.2, 0.3],
    [0.1, 0.2, -0.7, 0.5],
], dtype=torch.float32, requires_grad=True)
b = torch.tensor([0.1, -0.2, 0.05], dtype=torch.float32, requires_grad=True)
ln_w = torch.tensor([1.1, 0.9, 1.2], dtype=torch.float32, requires_grad=True)
ln_b = torch.tensor([0.05, -0.05, 0.02], dtype=torch.float32, requires_grad=True)
target = torch.tensor([2, 0], dtype=torch.long)

logits = F.layer_norm(F.linear(x, w, b), [3], ln_w, ln_b, 1e-5)
loss = F.cross_entropy(logits, target)
loss.backward()

# 2. Causal SDPA
q = torch.arange(1, 25, dtype=torch.float32).reshape(1, 2, 3, 4) * 0.1
k = torch.arange(24, 0, -1, dtype=torch.float32).reshape(1, 2, 3, 4) * 0.1
v = torch.arange(1, 25, dtype=torch.float32).reshape(1, 2, 3, 4) * 0.05
q.requires_grad_(True); k.requires_grad_(True); v.requires_grad_(True)
sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True)
sdpa_loss = sdpa.pow(2).mean()
sdpa_loss.backward()

# 3. Conv2d + RMSNorm
img = torch.arange(1, 33, dtype=torch.float32).reshape(1, 2, 4, 4) * 0.1
cw = torch.arange(1, 55, dtype=torch.float32).reshape(3, 2, 3, 3) * 0.02
cb = torch.tensor([0.1, -0.1, 0.2], dtype=torch.float32)
img.requires_grad_(True); cw.requires_grad_(True); cb.requires_grad_(True)
conv_out = F.conv2d(img, cw, cb, stride=1, padding=1)
rn_w = torch.tensor([1.0, 1.2, 0.8, 1.1], dtype=torch.float32, requires_grad=True)
rn_out = F.rms_norm(conv_out, [4], rn_w, 1e-5)
conv_loss = rn_out.sum()
conv_loss.backward()

# 4. Two steps of AdamW on a single parameter
p = torch.tensor([1.5, -2.0, 0.5], dtype=torch.float32, requires_grad=True)
opt = torch.optim.AdamW([p], lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05)
for step in range(2):
    opt.zero_grad()
    l = (p * torch.tensor([2.0, -1.0, 3.0]) - torch.tensor([1.0, 1.0, 1.0])).pow(2).sum()
    l.backward()
    opt.step()

print(json.dumps({
    "loss": loss.item(),
    "x_grad": x.grad.flatten().tolist(),
    "w_grad": w.grad.flatten().tolist(),
    "b_grad": b.grad.flatten().tolist(),
    "ln_w_grad": ln_w.grad.flatten().tolist(),
    "sdpa_out": sdpa.detach().flatten().tolist(),
    "q_grad": q.grad.flatten().tolist(),
    "conv_loss": conv_loss.item(),
    "cw_grad": cw.grad.flatten().tolist(),
    "rn_w_grad": rn_w.grad.flatten().tolist(),
    "adamw_p": p.detach().flatten().tolist(),
}))
`;

    const ref = JSON.parse(
      execFileSync(py, ['-c', pyScript], { encoding: 'utf8', env: pythonSubprocessEnv() })
    ) as {
      loss: number;
      x_grad: number[];
      w_grad: number[];
      b_grad: number[];
      ln_w_grad: number[];
      sdpa_out: number[];
      q_grad: number[];
      conv_loss: number;
      cw_grad: number[];
      rn_w_grad: number[];
      adamw_p: number[];
    };

    torch.tidy(() => {
      // 1. Linear + LayerNorm + CrossEntropy
      const x = torch.tensor(
        [
          [0.5, -1.2, 0.3, 2.1],
          [-0.4, 0.8, 1.5, -0.7],
        ],
        { dtype: torch.float32, requiresGrad: true }
      );
      const w = torch.tensor(
        [
          [0.2, -0.3, 0.4, 0.1],
          [-0.5, 0.6, -0.2, 0.3],
          [0.1, 0.2, -0.7, 0.5],
        ],
        { dtype: torch.float32, requiresGrad: true }
      );
      const b = torch.tensor([0.1, -0.2, 0.05], { dtype: torch.float32, requiresGrad: true });
      const lnW = torch.tensor([1.1, 0.9, 1.2], { dtype: torch.float32, requiresGrad: true });
      const lnB = torch.tensor([0.05, -0.05, 0.02], { dtype: torch.float32, requiresGrad: true });
      const target = torch.tensor([2, 0], { dtype: torch.int64 });

      const logits = F.layerNorm(F.linear(x, w, b), [3], lnW, lnB, 1e-5);
      const loss = F.crossEntropy(logits, target);
      loss.backward();

      assert.ok(Math.abs(loss.item() - ref.loss) < 1e-5);
      assertCloseArray(Array.from(x.grad!.toTypedArray() as Float32Array), ref.x_grad, 1e-5, 'x.grad');
      assertCloseArray(Array.from(w.grad!.toTypedArray() as Float32Array), ref.w_grad, 1e-5, 'w.grad');
      assertCloseArray(Array.from(b.grad!.toTypedArray() as Float32Array), ref.b_grad, 1e-5, 'b.grad');
      assertCloseArray(Array.from(lnW.grad!.toTypedArray() as Float32Array), ref.ln_w_grad, 1e-5, 'lnW.grad');

      // 2. Causal SDPA
      const q = torch.arange(1, 25, 1, { dtype: torch.float32 }).reshape(1, 2, 3, 4).mul(0.1).detach().requiresGrad_(true);
      const k = torch.arange(24, 0, -1, { dtype: torch.float32 }).reshape(1, 2, 3, 4).mul(0.1).detach().requiresGrad_(true);
      const v = torch.arange(1, 25, 1, { dtype: torch.float32 }).reshape(1, 2, 3, 4).mul(0.05).detach().requiresGrad_(true);
      const sdpa = F.scaledDotProductAttention(q, k, v, { isCausal: true });
      sdpa.pow(2).mean().backward();
      assertCloseArray(Array.from(sdpa.toTypedArray() as Float32Array), ref.sdpa_out, 1e-5, 'sdpa');
      assertCloseArray(Array.from(q.grad!.toTypedArray() as Float32Array), ref.q_grad, 1e-5, 'q.grad');

      // 3. Conv2d + RMSNorm
      const img = torch.arange(1, 33, 1, { dtype: torch.float32 }).reshape(1, 2, 4, 4).mul(0.1).detach().requiresGrad_(true);
      const cw = torch.arange(1, 55, 1, { dtype: torch.float32 }).reshape(3, 2, 3, 3).mul(0.02).detach().requiresGrad_(true);
      const cb = torch.tensor([0.1, -0.1, 0.2], { dtype: torch.float32, requiresGrad: true });
      const convOut = F.conv2d(img, cw, { bias: cb, stride: 1, padding: 1 });
      const rnW = torch.tensor([1.0, 1.2, 0.8, 1.1], { dtype: torch.float32, requiresGrad: true });
      const rnOut = F.rmsNorm(convOut, [4], rnW, 1e-5);
      const convLoss = rnOut.sum();
      convLoss.backward();
      assert.ok(Math.abs(convLoss.item() - ref.conv_loss) < 1e-4);
      assertCloseArray(Array.from(cw.grad!.toTypedArray() as Float32Array), ref.cw_grad, 1e-4, 'cw.grad');
      assertCloseArray(Array.from(rnW.grad!.toTypedArray() as Float32Array), ref.rn_w_grad, 1e-4, 'rnW.grad');

      // 4. Two steps of AdamW
      using p = new torch.nn.Parameter(torch.tensor([1.5, -2.0, 0.5], { dtype: torch.float32 }));
      using opt = new optim.AdamW([p], { lr: 0.1, betas: [0.9, 0.999], eps: 1e-8, weightDecay: 0.05 });
      const coeff = torch.tensor([2.0, -1.0, 3.0]);
      const ones = torch.tensor([1.0, 1.0, 1.0]);
      for (let step = 0; step < 2; step++) {
        opt.zeroGrad();
        const l = p.mul(coeff).sub(ones).pow(2).sum();
        l.backward();
        opt.step();
      }
      assertCloseArray(Array.from(p.toTypedArray() as Float32Array), ref.adamw_p, 1e-5, 'adamw.p');
    });
  });
});
