# Neural Network Implementation with Mini-Batch Gradient Descent
A TypeScript implementation of a neural network using mini-batch gradient descent optimization, featuring customizable activation functions, regularization techniques, and model persistence.
Mathematical Foundation
1. Network Architecture
The neural network is structured with multiple layers, where each layer l performs the following transformation:
$z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}$
Where:

$z^{(l)}$ is the weighted input to layer l
$W^{(l)}$ is the weight matrix for layer l
$a^{(l-1)}$ is the activation from the previous layer
$b^{(l)}$ is the bias vector for layer l

2. Activation Functions
The implementation supports multiple activation functions:
Sigmoid
$\sigma(x) = \frac{1}{1 + e^{-x}}$

Range: $(0,1)$
Used in hidden layers and binary classification output
Derivative: $\sigma(x)(1 - \sigma(x))$

ReLU (Rectified Linear Unit)
$f(x) = \max(0,x)$

Range: $[0,\infty)$
Helps prevent vanishing gradient problem
Derivative: $1$ if $x > 0$, $0$ otherwise

Softmax (for output layer)
$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$

Used for multi-class classification
Outputs sum to 1, representing probabilities

3. Loss Function and Regularization
Mean Squared Error (MSE)
$L = \frac{1}{n}\sum(y - \hat{y})^2$

$n$: number of samples
$y$: true value
$\hat{y}$: predicted value

Regularization Terms
L2 Regularization: $\frac{\lambda}{2m} \sum |W|^2$
L1 Regularization: $\frac{\lambda}{m} \sum |W|$

$\lambda$: regularization strength
$m$: mini-batch size
$W$: weights

## Implementation Details

### Project Structure
```
src/
├── ActivationFunction.ts
├── NeuralNetwork.ts
├── Regularization.ts
├── MiniBatch.ts
└── ModelStorage.ts
```

### Key Components

1. **ActivationFunction Class**
```typescript
export class ActivationFunction {
  static sigmoid(x: number): number;
  static relu(x: number): number;
  static tanh(x: number): number;
  static softmax(x: number[]): number[];
}
```

2. **NeuralNetwork Class**
```typescript
export class NeuralNetwork {
  static async compute_loss(data: NeuralNetworkParams, ...): number;
  static async compute_grad_W(data: NeuralNetworkParams, ...): number;
  static async compute_grad_B(data: NeuralNetworkParams, ...): number;
}
```

## Installation and Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/neural-network-implementation.git
cd neural-network-implementation
```

2. Install dependencies:
```bash
pnpm install
```

3. Build the project:
```bash
pnpm run build
```

## Usage Examples

### Basic Training

```typescript
import { NeuralNetwork, ActivationFunction, Regularization } from './src';

// Define training data
const input = [1.0, 2.0, 3.0, 4.0, 5.0];
const output = [1.5, 2.0, 3.1, 4.1, 5.6];

// Configure network
const networkConfig = {
  layers: [
    { size: 2, activation: 'sigmoid' },
    { size: 1, activation: 'linear' }
  ],
  learningRate: 0.01,
  regularization: {
    type: 'l2',
    lambda: 0.01
  }
};

// Train the model
const model = new NeuralNetwork(networkConfig);
await model.train(input, output, {
  epochs: 1000,
  batchSize: 32,
  verbose: true
});
```

### Model Persistence

```typescript
// Save model
await ModelStorage.saveModel(model, 'trained_model.json');

// Load model
const loadedModel = await ModelStorage.loadModel('trained_model.json');
```

## Advanced Features

### 1. Mini-Batch Processing
The `MiniBatch` class handles data splitting and iteration:

```typescript
const batchProcessor = new MiniBatch(input, output, 32);
batchProcessor.iterateBatches((batch) => {
  // Process each mini-batch
});
```

### 2. Custom Regularization
Implement custom regularization by extending the `Regularization` class:

```typescript
class CustomRegularization extends Regularization {
  static custom(weight: number, lambda: number): number {
    // Custom regularization logic
    return lambda * Math.pow(weight, 3);
  }
}
```

## Performance Optimization Tips

1. **Batch Size Selection**
   - Larger batches: Better gradient estimates, more memory
   - Smaller batches: Faster iterations, more noise

2. **Learning Rate Tuning**
   - Too high: Unstable training
   - Too low: Slow convergence
   - Recommended: Start with 0.01 and adjust

3. **Regularization Strength**
   - Increase λ to reduce overfitting
   - Decrease λ if underfitting

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
