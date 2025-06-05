# Deep Learning Application

This project is a deep learning framework implemented in C++. The goal is to provide a flexible and efficient library for building and training neural networks, with the potential to compete with established frameworks like PyTorch and TensorFlow.

## Project Structure

- **src/**: Contains the source code for the deep learning framework.
  - **core/**: Implements core functionalities such as tensor operations.
    - `tensor.cpp`: Implementation of the Tensor class.
    - `tensor.h`: Declaration of the Tensor class.
  - **layers/**: Implements various neural network layers.
    - `layer.cpp`: Implementation of the Layer class.
    - `layer.h`: Declaration of the Layer class.
  - **models/**: Implements the Model class for managing neural networks.
    - `model.cpp`: Implementation of the Model class.
  - **utils/**: Contains utility functions for data handling and performance measurement.
    - `utils.h`: Declaration of utility functions.
  - `main.cpp`: Entry point of the application.

- **include/**: Contains public headers for the framework.
  - `tensor.h`: Public interface for the Tensor class.
  - `layer.h`: Public interface for the Layer class.
  - `model.h`: Public interface for the Model class.
  - `utils.h`: Public interface for utility functions.

- **tests/**: Contains unit tests for the framework.
  - `test_main.cpp`: Unit tests for core functionalities.

- **CMakeLists.txt**: Configuration file for building the project with CMake.

## Recent Developments and Features

### Hybrid Activation Function Design
- The `Layer` class now supports a hybrid activation function approach:
  - You can specify a **single activation function** for all neurons in a layer (e.g., ReLU, sigmoid, etc.).
  - Alternatively, you can provide a **vector of activation functions**, one for each neuron, allowing for maximum flexibility and research use cases.
- This design makes the framework both scalable (for standard deep learning) and highly customizable (for advanced architectures).

### Improved Layer and Neuron APIs
- The `Layer` constructor now requires the number of neurons, the input size, and either a single activation function or a vector of activation functions.
- Each `Neuron` can be configured with custom weights and bias, and computes its output as a weighted sum plus bias.

### Expanded Test Coverage
- The test suite now includes:
  - **Tensor**: Addition and multiplication operations.
  - **Neuron**: Output calculation with ground truth validation.
  - **Layer**: Both single and per-neuron activation function scenarios are tested.
  - **Model**: Basic training interface is tested for integration.
- All tests pass, ensuring the correctness of the core components and the new hybrid activation design.

## Activation Function Design: Per-Neuron vs. Per-Layer

This framework supports a **hybrid approach** to activation functions, giving you both flexibility and modularity:

- **Per-neuron activation:**
  - Each neuron in a dense layer can have its own activation function.
  - Useful for research and experimentation with heterogeneous layers.
  - Example: `DenseLayer(3, 4, std::vector<std::function<float(float)>>{activations::ReLU, activations::Sigmoid, activations::Tanh})`

- **Per-layer activation (ActivationLayer):**
  - All neurons in a layer use the same activation function, applied as a separate layer.
  - Matches the design of most modern frameworks (PyTorch, TensorFlow, Keras).
  - More modular and efficient for standard use cases.
  - Example: `model.addLayer(new DenseLayer(...)); model.addLayer(new ActivationLayer(activations::ReLU));`

### Summary Table

| Approach                | Per-neuron flexibility | Standard/Modular | Performance |
|-------------------------|-----------------------|------------------|-------------|
| Neuron-level activation | Yes                   | No               | Lower       |
| Activation layer        | No (per layer only)   | Yes              | Higher      |
| Hybrid (this framework) | Yes                   | Yes              | Medium      |

### Recommendation
- For most users and production models, use per-layer activation (ActivationLayer) for clarity and performance.
- For research or custom architectures, use per-neuron activation in DenseLayer.
- You can mix and match both approaches as needed in your model definitions.

## Getting Started

### Prerequisites

- C++11 or later
- CMake

### Building the Project

1. Clone the repository:
   ```
   git clone <repository-url>
   cd deep-learning-app
   ```

2. Create a build directory and navigate into it:
   ```
   mkdir build
   cd build
   ```

3. Run CMake to configure the project:
   ```
   cmake ..
   ```

4. Build the project:
   ```
   make
   ```

### Usage

After building the project, you can run the application by executing the generated binary. You can modify `main.cpp` to set up your model and start training or inference.

### Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

### License

This project is licensed under the MIT License. See the LICENSE file for more details.