import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import torch, { F, nn, optim, utils } from '../src/index.js';

describe('End-to-end training runs (MLP, CNN, causal Transformer) & leak verification', () => {
  torch.setNumThreads(4);

  it('trains a nonlinear MLP with AdamW + CosineAnnealingLR and converges cleanly', () => {
    torch.manualSeed(1234);
    const baseline = torch.memoryStats();

    const { initialLoss, finalLoss } = (() => {
      using model = new nn.Sequential(
        new nn.Linear(2, 32),
        new nn.GELU(),
        new nn.Linear(32, 32),
        new nn.GELU(),
        new nn.Linear(32, 1)
      );
      using optimizer = new optim.AdamW(model.parameters(), { lr: 2e-2, weightDecay: 1e-4 });
      const scheduler = new optim.CosineAnnealingLR(optimizer, 120, 1e-3);
      const criterion = new nn.MSELoss();

      using x = torch.randn([128, 2]);
      using y = torch.tidy(() => {
        const x0 = x.at(':', 0);
        const x1 = x.at(':', 1);
        return x0.sin().add(x1.cos()).unsqueeze(1);
      });

      let first = 0;
      let last = 0;
      for (let epoch = 0; epoch < 120; epoch++) {
        const lossVal = torch.tidy(() => {
          optimizer.zeroGrad();
          const pred = model.forward(x);
          const loss = criterion.forward(pred, y);
          loss.backward();
          optimizer.step();
          return loss.item();
        });
        scheduler.step();
        if (epoch === 0) first = lossVal;
        last = lossVal;
      }
      return { initialLoss: first, finalLoss: last };
    })();

    assert.ok(finalLoss < initialLoss * 0.05, `Expected strong convergence: ${initialLoss} -> ${finalLoss}`);
    assert.ok(finalLoss < 0.02, `Expected final MSE < 0.02, got ${finalLoss}`);
    assert.equal(torch.memoryStats().liveTensors, baseline.liveTensors);
  });

  it('trains a 2D CNN classifier (Conv2d + BatchNorm2d + MaxPool2d) to >95% accuracy', () => {
    torch.manualSeed(7);

    class TinyCNN extends nn.Module {
      readonly conv1 = new nn.Conv2d(1, 8, 3, { padding: 1 });
      readonly bn1 = new nn.BatchNorm2d(8);
      readonly pool = new nn.MaxPool2d(2);
      readonly conv2 = new nn.Conv2d(8, 16, 3, { padding: 1 });
      readonly gap = new nn.AdaptiveAvgPool2d(1);
      readonly head = new nn.Linear(16, 2);

      forward(x: torch.Tensor): torch.Tensor {
        const h1 = this.pool.forward(this.bn1.forward(this.conv1.forward(x)).relu());
        const h2 = this.gap.forward(this.conv2.forward(h1).relu()).flatten(1);
        return this.head.forward(h2);
      }
    }

    using cnn = new TinyCNN();
    using opt = new optim.SGD(cnn.parameters(), { lr: 0.1, momentum: 0.9 });
    const lossFn = new nn.CrossEntropyLoss();

    // Synthetic dataset: class 0 has horizontal stripe, class 1 has vertical stripe + noise
    const { x, y } = torch.tidy(() => {
      const n = 64;
      const imgs = torch.randn([n, 1, 8, 8]).mul(0.25);
      const labels = torch.zeros([n], { dtype: torch.int64 });
      for (let i = 0; i < n; i++) {
        if (i % 2 === 0) {
          imgs.indexPut_([i, 0, 3, ':'], 2.0);
          labels.indexPut_([i], 0);
        } else {
          imgs.indexPut_([i, 0, ':', 4], 2.0);
          labels.indexPut_([i], 1);
        }
      }
      return { x: imgs, y: labels };
    });

    try {
      for (let step = 0; step < 35; step++) {
        torch.tidy(() => {
          opt.zeroGrad();
          const logits = cnn.forward(x);
          const loss = lossFn.forward(logits, y);
          loss.backward();
          opt.step();
        });
      }

      const acc = torch.noGrad(() =>
        torch.tidy(() => {
          cnn.eval();
          const preds = cnn.forward(x).argmax(1);
          return preds.eq(y).float().mean().item();
        })
      );
      assert.ok(acc >= 0.95, `Expected CNN accuracy >= 0.95, got ${acc}`);
    } finally {
      x.dispose();
      y.dispose();
    }
  });

  it('trains a mini causal Transformer LM (Embedding + SDPA TransformerBlock) and leaks zero tensors', () => {
    torch.manualSeed(42);
    const baseline = torch.memoryStats();

    class MiniGPT extends nn.Module {
      readonly tokEmb: nn.Embedding;
      readonly posEmb: nn.Embedding;
      readonly block: nn.TransformerBlock;
      readonly norm: nn.LayerNorm;
      readonly lmHead: nn.Linear;

      constructor(vocabSize: number, dModel: number, nHead: number, maxSeq: number) {
        super();
        this.tokEmb = new nn.Embedding(vocabSize, dModel);
        this.posEmb = new nn.Embedding(maxSeq, dModel);
        this.block = new nn.TransformerBlock(dModel, nHead, { dimFeedforward: dModel * 2 });
        this.norm = new nn.LayerNorm(dModel);
        this.lmHead = new nn.Linear(dModel, vocabSize, { bias: false });
      }

      forward(tokens: torch.Tensor): torch.Tensor {
        const T = tokens.size(1);
        const pos = torch.arange(T, { dtype: torch.int64, device: tokens.device });
        const x = this.tokEmb.forward(tokens).add(this.posEmb.forward(pos));
        const h = this.block.forward(x, { isCausal: true });
        return this.lmHead.forward(this.norm.forward(h));
      }
    }

    const { firstLoss, lastLoss } = (() => {
      const vocab = 16;
      using gpt = new MiniGPT(vocab, 32, 4, 16);
      using opt = new optim.AdamW(gpt.parameters(), { lr: 3e-2, weightDecay: 1e-2 });

      // Two structured token sequences to learn next-token prediction
      using seq = torch.tensor(
        [
          [1, 2, 3, 4, 5, 6, 7, 8, 9],
          [9, 8, 7, 6, 5, 4, 3, 2, 1],
          [2, 4, 6, 8, 10, 12, 14, 1, 3],
          [1, 3, 5, 7, 9, 11, 13, 15, 0],
        ],
        { dtype: torch.int64 }
      );
      using inputs = torch.tidy(() => seq.at(':', torch.slice(0, 8)).contiguous());
      using targets = torch.tidy(() => seq.at(':', torch.slice(1, 9)).contiguous());

      let first = 0;
      let last = 0;
      for (let step = 0; step < 60; step++) {
        const lossVal = torch.tidy(() => {
          opt.zeroGrad();
          const logits = gpt.forward(inputs); // [B, T, V]
          const loss = F.crossEntropy(logits.reshape(-1, vocab), targets.reshape(-1));
          loss.backward();
          utils.clipGradNorm_(gpt.parameters(), 1.0);
          opt.step();
          return loss.item();
        });
        if (step === 0) first = lossVal;
        last = lossVal;
      }
      return { firstLoss: first, lastLoss: last };
    })();

    assert.ok(lastLoss < 0.05, `Expected MiniGPT loss < 0.05, got ${firstLoss} -> ${lastLoss}`);
    assert.equal(torch.memoryStats().liveTensors, baseline.liveTensors);
    assert.equal(torch.memoryStats().reportedExternalBytes, baseline.reportedExternalBytes);
  });
});
