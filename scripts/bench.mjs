import { execFileSync } from 'node:child_process';
import { performance } from 'node:perf_hooks';
import torch, { F, nn, optim } from '../dist/src/index.js';
import { findLibTorchDir } from './find-libtorch.mjs';

torch.setNumThreads(4);

function bench(name, warmup, iters, fn) {
  for (let i = 0; i < warmup; i++) fn();
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) fn();
  const dt = performance.now() - t0;
  return { name, msPerIter: dt / iters };
}

const results = [];

// 1. Small op dispatch latency (N-API overhead)
{
  using a = torch.randn([4, 4]);
  using b = torch.randn([4, 4]);
  results.push(
    bench('add [4,4] (dispatch overhead)', 200, 5000, () => {
      using c = a.add(b);
    })
  );
}

// 2. Medium matmul [256, 256]
{
  using a = torch.randn([256, 256]);
  using b = torch.randn([256, 256]);
  results.push(
    bench('matmul [256,256]', 10, 100, () => {
      using c = a.matmul(b);
    })
  );
}

// 3. Transformer forward+backward+fused AdamW step
{
  using block = new nn.TransformerBlock(64, 4, { dimFeedforward: 256 });
  using head = new nn.Linear(64, 32);
  using opt = new optim.AdamW([...block.parameters(), ...head.parameters()], { lr: 1e-3, fused: true });
  using x = torch.randn([8, 32, 64]);
  using y = torch.randint(0, 32, [8, 32]);
  results.push(
    bench('TransformerBlock(d=64,B=8,T=32) fwd+bwd+fusedAdamW', 5, 40, () => {
      torch.tidy(() => {
        opt.zeroGrad();
        const logits = head.forward(block.forward(x, { isCausal: true }));
        const loss = F.crossEntropy(logits.reshape(-1, 32), y.reshape(-1));
        loss.backward();
        opt.step();
      });
    })
  );
}

// Compare against Python PyTorch on the exact same 4 CPU threads
const pyBin = process.env.NORCH_PYTHON || `${findLibTorchDir()}/../../../../bin/python`;
const pyScript = `
import json, time, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(4)

def bench(name, warmup, iters, fn):
    for _ in range(warmup): fn()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    return {"name": name, "msPerIter": (time.perf_counter() - t0) * 1000.0 / iters}

out = []
a, b = torch.randn(4, 4), torch.randn(4, 4)
out.append(bench("add [4,4] (dispatch overhead)", 200, 5000, lambda: a.add(b)))

a2, b2 = torch.randn(256, 256), torch.randn(256, 256)
out.append(bench("matmul [256,256]", 10, 100, lambda: a2.matmul(b2)))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(64)
        self.q = nn.Linear(64, 64); self.k = nn.Linear(64, 64); self.v = nn.Linear(64, 64); self.o = nn.Linear(64, 64)
        self.fc1 = nn.Linear(64, 256); self.fc2 = nn.Linear(256, 64); self.head = nn.Linear(64, 32)
    def forward(self, x):
        B, T, D = x.shape
        h = self.norm1(x)
        q = self.q(h).view(B, T, 4, 16).transpose(1, 2)
        k = self.k(h).view(B, T, 4, 16).transpose(1, 2)
        v = self.v(h).view(B, T, 4, 16).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).contiguous().view(B, T, D)
        r1 = x + self.o(attn)
        r2 = r1 + self.fc2(F.gelu(self.fc1(self.norm2(r1))))
        return self.head(r2)

m = Block()
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, fused=True)
x = torch.randn(8, 32, 64)
y = torch.randint(0, 32, (8, 32))
def step():
    opt.zero_grad()
    loss = F.cross_entropy(m(x).reshape(-1, 32), y.reshape(-1))
    loss.backward()
    opt.step()

out.append(bench("TransformerBlock(d=64,B=8,T=32) fwd+bwd+fusedAdamW", 5, 40, step))
print(json.dumps(out))
`;

let pyResults = [];
try {
  pyResults = JSON.parse(execFileSync(pyBin, ['-c', pyScript], { encoding: 'utf8', timeout: 60000 }));
} catch (err) {
  console.error('Python benchmark skipped:', err.message);
}

console.log('\nCPU Benchmark (4 threads, libtorch ' + torch.libtorchVersion() + '):');
console.log('Benchmark'.padEnd(55) + 'norch (ms)'.padStart(14) + 'PyTorch (ms)'.padStart(14) + 'Ratio'.padStart(10));
console.log('-'.repeat(93));
for (let i = 0; i < results.length; i++) {
  const n = results[i];
  const p = pyResults[i];
  const ratio = p ? `${(n.msPerIter / p.msPerIter).toFixed(2)}x` : 'n/a';
  console.log(
    n.name.padEnd(55) +
      n.msPerIter.toFixed(4).padStart(14) +
      (p ? p.msPerIter.toFixed(4) : 'n/a').padStart(14) +
      ratio.padStart(10)
  );
}
