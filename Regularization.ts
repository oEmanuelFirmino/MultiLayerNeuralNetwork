export class Regularization {
  static l2(weight: number, lambda: number): number {
    return lambda * weight ** 2;
  }

  static l1(weight: number, lambda: number): number {
    return lambda * Math.abs(weight);
  }
}
