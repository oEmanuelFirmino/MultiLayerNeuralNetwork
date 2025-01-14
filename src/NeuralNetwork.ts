import { NeuralNetworkParams } from "./NeuralNetworkParams.interface";

export class NeuralNetwork {
  static async compute_loss(
    data: NeuralNetworkParams,
    activation: (x: number) => number,
    lambda: number,
    regularization: (weight: number, lambda: number) => number,
  ) {
    let loss: number = 0.0;
    const { input, output, weight, bias } = data;
    const n = input.length;

    for (let i = 0; i < n; i++) {
      let prediction: number = weight * output[i] + bias;
      prediction = activation(prediction);
      loss += (input[i] - prediction) ** 2;
    }


    return (loss + regularization(weight, lambda)) / n;
  }

  static async compute_grad_W(
    data: NeuralNetworkParams,
    activation: (x: number) => number,
    regularization: (weight: number, lambda: number) => number,
    lambda: number
  ) {
    let grad_W: number = 0.0;
    const { input, output, weight, bias } = data;
    const n = input.length;

    for (let i = 0; i < n; i++) {
      let prediction: number = weight * output[i] + bias;
      prediction = activation(prediction);
      grad_W += 2 * output[i] * (input[i] - prediction);
    }

    return (grad_W += regularization(weight, lambda)) / n;
  }

  static async compute_grad_B(
    data: NeuralNetworkParams,
    activation: (x: number) => number
  ) {
    let grad_B: number = 0.0;
    const { input, output, weight, bias } = data;
    const n = input.length;

    for (let i = 0; i < n; i++) {
      let prediction: number = weight * output[i] + bias;
      prediction = activation(prediction);
      grad_B += 2 * (input[i] - prediction);
    }

    return grad_B / n;
  }
}
