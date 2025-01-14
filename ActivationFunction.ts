export class ActivationFunction {
  static sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  };

  static relu(x: number): number {
    return Math.max(0, x);
  };

  static tanh(x: number): number {
    return Math.tanh(x);
  };

  static sigmoid_derivative(x: number): number {
    return x * (1 - x);
  };

  static softmax(x: number[]): number[] {
    const max = Math.max(...x);
    const exps = x.map(val => Math.exp(val - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(val => val / sum);
  }
}