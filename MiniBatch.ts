// MiniBatch.ts
export class MiniBatch {
  private input: number[];
  private output: number[];
  private miniBatchSize: number;
  private numMiniBatches: number;

  constructor(input: number[], output: number[], miniBatchSize: number) {
    this.input = input;
    this.output = output;
    this.miniBatchSize = miniBatchSize;
    this.numMiniBatches = Math.floor(input.length / miniBatchSize);
  }

  getNextBatch(batchIndex: number) {
    const start = batchIndex * this.miniBatchSize;
    const end = Math.min((batchIndex + 1) * this.miniBatchSize, this.input.length);

    return {
      input: this.input.slice(start, end),
      output: this.output.slice(start, end),
    };
  }

  iterateBatches(callback: (miniBatch: { input: number[]; output: number[] }) => void) {
    for (let i = 0; i < this.numMiniBatches; i++) {
      const miniBatch = this.getNextBatch(i);
      callback(miniBatch);
    }
  }
}
