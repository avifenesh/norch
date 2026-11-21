import assert from 'node:assert/strict';
import torch, { distributed as dist, nn, optim, utils } from '../../src/index.js';

function toNums(t: torch.Tensor): number[] {
  return Array.from(t.toTypedArray() as ArrayLike<number | bigint>, Number);
}

const rank = Number(process.env.RANK);
const worldSize = Number(process.env.WORLD_SIZE);
const initMethod = process.env.NORCH_INIT_METHOD ?? 'env://';
const mode = process.env.NORCH_DIST_MODE ?? 'full';

const pg = dist.initProcessGroup({
  backend: 'gloo',
  initMethod,
  rank,
  worldSize,
  timeoutMs: 30_000,
});

try {
  assert.equal(dist.isInitialized(), true);
  assert.equal(dist.getRank(), rank);
  assert.equal(dist.getWorldSize(), worldSize);
  assert.equal(dist.getBackend(), 'gloo');

  if (mode === 'filestore') {
    using t = torch.tensor([rank + 1, (rank + 1) * 10], { dtype: 'float32' });
    dist.allReduce(t, dist.ReduceOp.SUM);
    assert.deepEqual(toNums(t), [3, 30]);
    dist.barrier();
    console.log(JSON.stringify({ ok: true, rank, sum: toNums(t) }));
  } else {
    // 1. Store & barrier
    if (rank === 0) pg.storeSet('greeting', 'hello-norch');
    dist.barrier();
    assert.equal(pg.storeGet('greeting'), 'hello-norch');

    // 2. Broadcast
    using bcast = rank === 0 ? torch.tensor([1.5, 2.5, 3.5]) : torch.zeros([3]);
    dist.broadcast(bcast, 0);
    assert.deepEqual(toNums(bcast), [1.5, 2.5, 3.5]);

    // 3. AllReduce (SUM, AVG, MAX, MIN, PRODUCT)
    using arSum = torch.tensor([rank + 1, (rank + 1) * 2], { dtype: 'float32' });
    dist.allReduce(arSum, dist.ReduceOp.SUM);
    assert.deepEqual(toNums(arSum), [3, 6]);

    using arAvg = torch.tensor([rank === 0 ? 2 : 6, rank === 0 ? 4 : 8], { dtype: 'float32' });
    dist.allReduce(arAvg, dist.ReduceOp.AVG);
    assert.deepEqual(toNums(arAvg), [4, 6]);

    using arMax = torch.tensor([rank === 0 ? 10 : 3, rank === 0 ? 2 : 9], { dtype: 'float32' });
    dist.allReduce(arMax, dist.ReduceOp.MAX);
    assert.deepEqual(toNums(arMax), [10, 9]);

    using arMin = torch.tensor([rank === 0 ? 10 : 3, rank === 0 ? 2 : 9], { dtype: 'float32' });
    dist.allReduce(arMin, dist.ReduceOp.MIN);
    assert.deepEqual(toNums(arMin), [3, 2]);

    using arProd = torch.tensor([rank === 0 ? 2 : 5, rank === 0 ? 3 : 4], { dtype: 'float32' });
    dist.allReduce(arProd, dist.ReduceOp.PRODUCT);
    assert.deepEqual(toNums(arProd), [10, 12]);

    // 4. AllGather & AllGatherIntoTensor
    using local = torch.tensor([rank * 10 + 1, rank * 10 + 2], { dtype: 'float32' });
    using g0 = torch.zeros([2], { dtype: 'float32' });
    using g1 = torch.zeros([2], { dtype: 'float32' });
    dist.allGather([g0, g1], local);
    assert.deepEqual(toNums(g0), [1, 2]);
    assert.deepEqual(toNums(g1), [11, 12]);

    using flatOut = torch.zeros([4], { dtype: 'float32' });
    dist.allGatherIntoTensor(flatOut, local);
    assert.deepEqual(toNums(flatOut), [1, 2, 11, 12]);

    // 5. ReduceScatter
    using rsIn0 = torch.tensor([rank + 1, rank + 2], { dtype: 'float32' });
    using rsIn1 = torch.tensor([(rank + 1) * 10, (rank + 2) * 10], { dtype: 'float32' });
    using rsOut = torch.zeros([2], { dtype: 'float32' });
    dist.reduceScatter(rsOut, [rsIn0, rsIn1], dist.ReduceOp.SUM);
    if (rank === 0) {
      assert.deepEqual(toNums(rsOut), [3, 5]);
    } else {
      assert.deepEqual(toNums(rsOut), [30, 50]);
    }

    // 6. Point-to-point send / recv
    if (rank === 0) {
      using msg = torch.tensor([42, 99], { dtype: 'float32' });
      dist.send(msg, 1, undefined, 7);
    } else {
      using buf = torch.zeros([2], { dtype: 'float32' });
      dist.recv(buf, 0, undefined, 7);
      assert.deepEqual(toNums(buf), [42, 99]);
    }
    dist.barrier();

    // 6b. Gather, Scatter, AllToAll, asyncOp DistWork, and shared-store newGroup subgroup
    using gIn = torch.tensor([rank + 10, rank + 20], { dtype: 'float32' });
    if (rank === 0) {
      using gOut0 = torch.zeros([2], { dtype: 'float32' });
      using gOut1 = torch.zeros([2], { dtype: 'float32' });
      dist.gather(gIn, [gOut0, gOut1], 0);
      assert.deepEqual(toNums(gOut0), [10, 20]);
      assert.deepEqual(toNums(gOut1), [11, 21]);
    } else {
      dist.gather(gIn, null, 0);
    }

    using sOut = torch.zeros([2], { dtype: 'float32' });
    if (rank === 0) {
      using sIn0 = torch.tensor([100, 101], { dtype: 'float32' });
      using sIn1 = torch.tensor([200, 201], { dtype: 'float32' });
      dist.scatter(sOut, [sIn0, sIn1], 0);
      assert.deepEqual(toNums(sOut), [100, 101]);
    } else {
      dist.scatter(sOut, null, 0);
      assert.deepEqual(toNums(sOut), [200, 201]);
    }

    using a2aIn0 = torch.tensor([rank * 10 + 1], { dtype: 'float32' });
    using a2aIn1 = torch.tensor([rank * 10 + 2], { dtype: 'float32' });
    using a2aOut0 = torch.zeros([1], { dtype: 'float32' });
    using a2aOut1 = torch.zeros([1], { dtype: 'float32' });
    const a2aWork = dist.allToAll([a2aOut0, a2aOut1], [a2aIn0, a2aIn1], undefined, true);
    assert.ok(a2aWork !== undefined);
    assert.equal(a2aWork!.wait(), true);
    assert.equal(a2aWork!.isCompleted(), true);
    assert.deepEqual(toNums(a2aOut0), [rank === 0 ? 1 : 2]);
    assert.deepEqual(toNums(a2aOut1), [rank === 0 ? 11 : 12]);

    // Shared-store subgroup via newGroup (no extra TCP port)
    const subPg = dist.newGroup([0, 1], { groupName: 'sub_test' });
    assert.ok(subPg !== null);
    using subT = torch.tensor([rank + 5], { dtype: 'float32' });
    const subWork = dist.allReduce(subT, dist.ReduceOp.SUM, subPg!, true);
    subWork!.wait();
    assert.deepEqual(toNums(subT), [11]);
    subPg!.destroy();

    // SyncBatchNorm computes identical global stats across ranks
    using sbn = new nn.SyncBatchNorm(2);
    using sbnIn =
      rank === 0
        ? torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        : torch.tensor([[5.0, 6.0], [7.0, 8.0]]);
    using sbnOut = sbn.forward(sbnIn);
    // Global mean = [4, 5], runningMean updated with momentum=0.1 -> 0 + 0.1*[4, 5] = [0.4, 0.5]
    const rm = toNums(sbn.runningMean!);
    assert.ok(Math.abs(rm[0]! - 0.4) < 1e-5 && Math.abs(rm[1]! - 0.5) < 1e-5);
    dist.barrier();

    // 7. DistributedSampler determinism & partitioning
    const sampler = new utils.DistributedSampler({ length: 8 }, { shuffle: true, seed: 123 });
    const ep0 = Array.from(sampler);
    sampler.setEpoch(1);
    const ep1 = Array.from(sampler);
    assert.equal(ep0.length, 4);
    assert.equal(ep1.length, 4);

    // 8. DDP training parity with single-process full-batch reference
    class TinyMLP extends nn.Module {
      readonly fc1 = new nn.Linear(4, 8);
      readonly fc2 = new nn.Linear(8, 2);
      forward(x: torch.Tensor): torch.Tensor {
        return this.fc2.forward(this.fc1.forward(x).tanh());
      }
    }

    // Deliberately different seeds per rank; DDP must broadcast rank-0 weights!
    // Use small bucketCapMb to force multiple coalesced async gradient buckets!
    torch.manualSeed(rank === 0 ? 777 : 999);
    using baseModel = new TinyMLP();
    using ddp = new nn.parallel.DistributedDataParallel(baseModel, { bucketCapMb: 0.0001 });
    assert.ok(ddp.bucketCount >= 2);
    using optimizer = new optim.SGD(ddp.parameters(), { lr: 0.1 });

    // Deterministic synthetic 8-sample dataset split evenly across rank 0 and rank 1
    torch.manualSeed(2026);
    using fullX = torch.randn([8, 4]);
    using fullY = torch.randn([8, 2]);
    using localX = rank === 0 ? fullX.narrow(0, 0, 4).clone() : fullX.narrow(0, 4, 4).clone();
    using localY = rank === 0 ? fullY.narrow(0, 0, 4).clone() : fullY.narrow(0, 4, 4).clone();

    // Verify noSync skips communication without deadlocking
    ddp.noSync(() => {
      torch.tidy(() => {
        optimizer.zeroGrad();
        const pred = ddp.forward(localX) as torch.Tensor;
        const loss = torch.F.mseLoss(pred, localY);
        loss.backward();
        optimizer.zeroGrad();
      });
    });
    dist.barrier();

    for (let step = 0; step < 5; step++) {
      torch.tidy(() => {
        optimizer.zeroGrad();
        const pred = ddp.forward(localX) as torch.Tensor;
        const loss = torch.F.mseLoss(pred, localY);
        loss.backward();
        optimizer.step();
      });
    }

    const weights = {
      fc1W: toNums(baseModel.fc1.weight),
      fc1B: toNums(baseModel.fc1.bias!),
      fc2W: toNums(baseModel.fc2.weight),
      fc2B: toNums(baseModel.fc2.bias!),
    };

    dist.barrier();
    console.log(JSON.stringify({ ok: true, rank, ep0, ep1, weights }));
  }
} finally {
  dist.destroyProcessGroup();
}
