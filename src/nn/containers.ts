import { Tensor } from '../tensor.js';
import { Module } from './module.js';

export class Sequential extends Module {
  private readonly layers: Module[] = [];

  constructor(...modules: Module[]) {
    super();
    modules.forEach((mod, i) => {
      this.layers.push(mod);
      this.addModule(String(i), mod);
    });
  }

  get length(): number {
    return this.layers.length;
  }

  at(index: number): Module | undefined {
    return this.layers.at(index);
  }

  forward(input: Tensor): Tensor {
    let x = input;
    for (const layer of this.layers) {
      x = layer.call(x) as Tensor;
    }
    return x;
  }
}

export class ModuleList<T extends Module = Module> extends Module implements Iterable<T> {
  private readonly items: T[] = [];

  constructor(modules: Iterable<T> = []) {
    super();
    for (const mod of modules) {
      this.push(mod);
    }
  }

  push(mod: T): this {
    const idx = this.items.length;
    this.items.push(mod);
    this.addModule(String(idx), mod);
    return this;
  }

  get length(): number {
    return this.items.length;
  }

  get(index: number): T {
    const item = this.items[index];
    if (!item) throw new RangeError(`ModuleList index out of bounds: ${index}`);
    return item;
  }

  [Symbol.iterator](): Iterator<T> {
    return this.items[Symbol.iterator]();
  }

  forward(): never {
    throw new Error('ModuleList does not implement forward(); iterate over it directly');
  }
}

export class ModuleDict<T extends Module = Module> extends Module {
  private readonly entries = new Map<string, T>();

  constructor(initial?: Record<string, T>) {
    super();
    if (initial) {
      for (const [k, v] of Object.entries(initial)) {
        this.set(k, v);
      }
    }
  }

  set(key: string, mod: T): this {
    this.entries.set(key, mod);
    this.addModule(key, mod);
    return this;
  }

  get(key: string): T | undefined {
    return this.entries.get(key);
  }

  keys(): IterableIterator<string> {
    return this.entries.keys();
  }

  values(): IterableIterator<T> {
    return this.entries.values();
  }

  forward(): never {
    throw new Error('ModuleDict does not implement forward()');
  }
}
