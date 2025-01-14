import { NeuralNetwork } from "./NeuralNetwork";
import { NeuralNetworkParams } from "./NeuralNetworkParams.interface";
import { ActivationFunction } from "./ActivationFunction";
import { Regularization } from "./Regularization";
import { MiniBatch } from "./MiniBatch";

const input: number[] = [1.0, 2.0, 3.0, 4.0, 5.0];
const output: number[] = [1.5, 2.0, 3.1, 4.1, 5.6];
let weight: number = 0.0;
let bias: number = 0.0;

const data: NeuralNetworkParams = {
  input,
  output,
  weight,
  bias
};

const eta: number = 0.0001;
const epochs: number = 1000;
const lambda: number = 0.01;
const miniBatchSize: number = 2;

const miniBatch = new MiniBatch(input, output, miniBatchSize);

const main = async (): Promise<number> => {
  console.log(`Data used: ${JSON.stringify(data)}`);

  for (let epoch: number = 0; epoch < epochs; epoch++) {
    miniBatch.iterateBatches(async (miniBatchData) => {

      const data: NeuralNetworkParams = {
        input: miniBatchData.input,
        output: miniBatchData.output,
        weight,
        bias
      }

      const grad_W = await NeuralNetwork.compute_grad_W(data, ActivationFunction.relu, Regularization.l1, lambda);
      const grad_B = await NeuralNetwork.compute_grad_B(data, ActivationFunction.relu);

      weight -= eta * grad_W;
      bias -= eta * grad_B;
    });

    if (epoch % 10 === 0) {
      const loss = await NeuralNetwork.compute_loss(data, ActivationFunction.relu, lambda, Regularization.l1);
      console.log(`Epoch: ${epoch}, Loss: ${loss}, Bias: ${bias}, Weight: ${weight}`);
    }
  }

  console.log(`Final Bias: ${bias}, Final Weight: ${weight}`);
  return 0;
}

main();
