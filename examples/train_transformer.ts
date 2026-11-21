import torch, { F, nn, optim, utils } from '../src/index.js';

torch.setNumThreads(4);

class MiniTransformerLM extends nn.Module {
  readonly tokEmb: nn.Embedding;
  readonly posEmb: nn.Embedding;
  readonly block: nn.TransformerBlock;
  readonly norm: nn.LayerNorm;
  readonly head: nn.Linear;

  constructor(vocabSize: number, dModel: number, nHead: number, maxSeq: number) {
    super();
    this.tokEmb = new nn.Embedding(vocabSize, dModel);
    this.posEmb = new nn.Embedding(maxSeq, dModel);
    this.block = new nn.TransformerBlock(dModel, nHead, { dimFeedforward: dModel * 4 });
    this.norm = new nn.LayerNorm(dModel);
    this.head = new nn.Linear(dModel, vocabSize, { bias: false });
  }

  forward(tokens: torch.Tensor): torch.Tensor {
    const seqLen = tokens.size(1);
    const pos = torch.arange(seqLen, { dtype: torch.int64, device: tokens.device });
    const h = this.tokEmb.forward(tokens).add(this.posEmb.forward(pos));
    return this.head.forward(this.norm.forward(this.block.forward(h, { isCausal: true })));
  }
}

torch.manualSeed(42);
const vocab = 32;
using model = new MiniTransformerLM(vocab, 64, 4, 16);
using optimizer = new optim.AdamW(model.parameters(), { lr: 2e-2, weightDecay: 1e-2 });
const scheduler = new optim.CosineAnnealingLR(optimizer, 50, 1e-3);

using data = torch.tensor(
  [
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
    [9, 8, 7, 6, 5, 4, 3, 2, 1],
    [2, 4, 6, 8, 10, 12, 14, 16, 18],
    [3, 6, 9, 12, 15, 18, 21, 24, 27],
  ],
  { dtype: torch.int64 }
);
using inputs = data.at(':', torch.slice(0, 8)).contiguous();
using targets = data.at(':', torch.slice(1, 9)).contiguous();

for (let step = 0; step < 50; step++) {
  const loss = torch.tidy(() => {
    optimizer.zeroGrad();
    const logits = model.forward(inputs);
    const l = F.crossEntropy(logits.reshape(-1, vocab), targets.reshape(-1));
    l.backward();
    utils.clipGradNorm_(model.parameters(), 1.0);
    optimizer.step();
    return l.item();
  });
  scheduler.step();
  if (step % 10 === 0 || step === 49) {
    console.log(`step=${step} loss=${loss.toFixed(4)} lr=${scheduler.getLastLr()[0]!.toFixed(5)}`);
  }
}
