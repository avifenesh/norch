import { execFileSync } from 'node:child_process';
import { performance } from 'node:perf_hooks';
import torch, { F, nn, optim } from '../dist/src/index.js';
import { findLibTorchDir } from './find-libtorch.mjs';

if (!torch.cuda.isAvailable()) {
  console.error('[norch] CUDA is not available on this host.');
  process.exit(1);
}

const props = torch.cuda.deviceProperties(0);
console.log(
  `[norch] GPU: ${props.name} (sm_${props.major}${props.minor}, ${(props.totalMemory / 1024 ** 3).toFixed(1)} GiB), libtorch ${torch.libtorchVersion()}`
);

function benchCuda(name, warmup, iters, fn) {
  for (let i = 0; i < warmup; i++) fn();
  torch.cuda.synchronize();
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) fn();
  torch.cuda.synchronize();
  const dt = performance.now() - t0;
  return { name, msPerIter: dt / iters };
}

const results = [];

// 1. BF16 Matmul [2048, 2048]
{
  using a = torch.randn([2048, 2048], { dtype: 'bfloat16', device: 'cuda' });
  using b = torch.randn([2048, 2048], { dtype: 'bfloat16', device: 'cuda' });
  results.push(
    benchCuda('matmul [2048,2048] bf16', 10, 50, () => {
      using c = a.matmul(b);
    })
  );
}

// 2. TransformerBlock(d=512, nhead=8, B=16, T=128) BF16 autocast fwd+bwd+fusedAdamW
{
  using block = new nn.TransformerBlock(512, 8, { dimFeedforward: 2048 });
  using head = new nn.Linear(512, 1024);
  block.to('cuda');
  head.to('cuda');
  using opt = new optim.AdamW([...block.parameters(), ...head.parameters()], { lr: 1e-3, fused: true });
  using x = torch.randn([16, 128, 512], { device: 'cuda' });
  using y = torch.randint(0, 1024, [16, 128], { device: 'cuda' });

  results.push(
    benchCuda('TransformerBlock(d=512,B=16,T=128) bf16 fwd+bwd+fusedAdamW', 5, 30, () => {
      torch.tidy(() => {
        opt.zeroGrad();
        const logits = torch.autocast('cuda', torch.bfloat16, () =>
          head.forward(block.forward(x, { isCausal: true }))
        );
        const loss = F.crossEntropy(logits.float().reshape(-1, 1024), y.reshape(-1));
        loss.backward();
        opt.step();
      });
    })
  );
}

// 3. CUDAGraph replay latency (chain of 8 ops on [1024, 1024])
{
  using inp = torch.randn([1024, 1024], { dtype: 'bfloat16', device: 'cuda' });
  using out = torch.zeros([1024, 1024], { dtype: 'bfloat16', device: 'cuda' });
  const runBody = () => {
    torch.tidy(() => {
      let h = inp;
      for (let k = 0; k < 8; k++) {
        h = h.mul(0.99).add(0.01).relu();
      }
      out.copy_(h);
    });
  };
  runBody();
  torch.cuda.synchronize();
  const g = new torch.cuda.CUDAGraph();
  torch.cuda.graph(g, runBody);
  results.push(
    benchCuda('CUDAGraph replay (8 ops [1024,1024] bf16)', 10, 200, () => {
      g.replay();
    })
  );
  g.reset();
}

torch.cuda.emptyCache();

// Compare against Python PyTorch on the exact same GPU
const pyBin = process.env.NORCH_PYTHON || `${findLibTorchDir()}/../../../../bin/python`;
const pyScript = `
import json, time, torch, torch.nn as nn, torch.nn.functional as F

def bench_cuda(name, warmup, iters, fn):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return {"name": name, "msPerIter": (time.perf_counter() - t0) * 1000.0 / iters}

out = []
a = torch.randn(2048, 2048, dtype=torch.bfloat16, device="cuda")
b = torch.randn(2048, 2048, dtype=torch.bfloat16, device="cuda")
out.append(bench_cuda("matmul [2048,2048] bf16", 10, 50, lambda: a @ b))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.q = nn.Linear(512, 512); self.k = nn.Linear(512, 512)
        self.v = nn.Linear(512, 512); self.o = nn.Linear(512, 512)
        self.fc1 = nn.Linear(512, 2048); self.fc2 = nn.Linear(2048, 512)
        self.head = nn.Linear(512, 1024)
    def forward(self, x):
        B, T, D = x.shape
        h = self.norm1(x)
        q = self.q(h).view(B, T, 8, 64).transpose(1, 2)
        k = self.k(h).view(B, T, 8, 64).transpose(1, 2)
        v = self.v(h).view(B, T, 8, 64).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).contiguous().view(B, T, D)
        r1 = x + self.o(attn)
        r2 = r1 + self.fc2(F.gelu(self.fc1(self.norm2(r1))))
        return self.head(r2)

m = Block().cuda()
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, fused=True)
x = torch.randn(16, 128, 512, device="cuda")
y = torch.randint(0, 1024, (16, 128), device="cuda")
def step():
    opt.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = m(x)
    loss = F.cross_entropy(logits.float().reshape(-1, 1024), y.reshape(-1))
    loss.backward()
    opt.step()

out.append(bench_cuda("TransformerBlock(d=512,B=16,T=128) bf16 fwd+bwd+fusedAdamW", 5, 30, step))

inp = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
buf = torch.zeros(1024, 1024, dtype=torch.bfloat16, device="cuda")
def body():
    h = inp
    for _ in range(8):
        h = torch.relu(h * 0.99 + 0.01)
    buf.copy_(h)
body()
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    body()
out.append(bench_cuda("CUDAGraph replay (8 ops [1024,1024] bf16)", 10, 200, g.replay))
g.reset()

print(json.dumps(out))
`;

let pyResults = [];
try {
  pyResults = JSON.parse(execFileSync(pyBin, ['-c', pyScript], { encoding: 'utf8', timeout: 120000 }));
} catch (err) {
  console.error('Python GPU benchmark skipped:', err.message);
}

console.log('\nGPU Benchmark (' + props.name + ', libtorch ' + torch.libtorchVersion() + '):');
console.log('Benchmark'.padEnd(62) + 'norch (ms)'.padStart(14) + 'PyTorch (ms)'.padStart(14) + 'Ratio'.padStart(10));
console.log('-'.repeat(100));
for (let i = 0; i < results.length; i++) {
  const n = results[i];
  const p = pyResults[i];
  const ratio = p ? `${(n.msPerIter / p.msPerIter).toFixed(2)}x` : 'n/a';
  console.log(
    n.name.padEnd(62) +
      n.msPerIter.toFixed(4).padStart(14) +
      (p ? p.msPerIter.toFixed(4) : 'n/a').padStart(14) +
      ratio.padStart(10)
  );
}
