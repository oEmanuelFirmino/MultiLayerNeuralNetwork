# Neural Network Training with Mini-Batch Gradient Descent

This project implements a simple neural network with mini-batch gradient descent and includes functionality to save/load trained models as JSON.

## Features

- Customizable multi-layer neural network.
- Mini-batch gradient descent for efficient training.
- Save and load models using JSON files.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/neural-network-training.git
   cd neural-network-training
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Train the neural network:
   ```bash
   npx ts-node main.ts
   ```

## Example Usage

### Training
The `main.ts` file configures and trains the model. Example:

```typescript
const input = [1.0, 2.0, 3.0, 4.0, 5.0];
const output = [1.5, 2.0, 3.1, 4.1, 5.6];

const layers = [
  { weights: [0.1, 0.2], biases: [0.1, 0.2] },
  { weights: [0.3, 0.4], biases: [0.3, 0.4] }
];

const nn = new NeuralNetwork(layers, ActivationFunction.sigmoid, Regularization.l2, 0.01);
n.train({ input, output, layers }, 1000, 0.01, 2);

ModelStorage.saveModel(nn.getLayers(), 'model.json');
```

### Saving and Loading Models
- Save a model:
  ```typescript
  ModelStorage.saveModel(layers, 'model.json');
  ```
- Load a model:
  ```typescript
  const loadedLayers = ModelStorage.loadModel('model.json');
  ```

## Output Format
Saved models are stored as JSON, e.g.:

```json
[
  { "weights": [0.1, 0.2], "biases": [0.1, 0.2] },
  { "weights": [0.3, 0.4], "biases": [0.3, 0.4] }
]
```

## License

This project is licensed under the MIT License.
