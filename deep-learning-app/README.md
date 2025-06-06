# Deep Learning Application

A flexible, extensible deep learning framework in C++ supporting advanced neuron types, heterogeneous layers, hybrid activation systems, and backend abstraction (CPU/GPU). Designed for both research and practical use.

---

## Project Structure

- **src/**: Source code for the framework
  - **core/**: Core components (e.g., `tensor.cpp`, `tensor.h`, `neuron.cpp`, `neuron.h`)
  - **layers/**: Layer implementations (e.g., `layer.cpp`, `layer.h`)
  - **models/**: Model management (`model.cpp`, `model.h`)
  - **utils/**: Utilities (`utils.h`)
  - `main.cpp`: Application entry point
- **include/**: Public headers
- **tests/**: Unit tests (`test_main.cpp`)
- **CMakeLists.txt**: Build configuration

---

## Key Features

### 1. **Polymorphic Neuron Types**
- **Supported neurons:**
  - `LinearNeuron`: Standard fully connected neuron
  - `QuadraticNeuron`: Quadratic form neuron
  - `SirenNeuron`: SIREN (sinusoidal representation networks)
  - `RBFNeuron`: Radial basis function neuron
  - `RationalNeuron`: Rational activation neuron
  - `ComplexNeuron`: Complex-valued neuron
- **Extensible:** Add your own neuron types by inheriting from the `Neuron` base class.

### 2. **Heterogeneous Layers**
- **Mix neuron types:**  
  Layers can contain any combination of neuron types, enabling research into heterogeneous architectures.
- **Example:**
  ```cpp
  std::vector<std::unique_ptr<Neuron>> neurons;
  neurons.push_back(std::make_unique<LinearNeuron>(...));
  neurons.push_back(std::make_unique<QuadraticNeuron>(...));
  // ...add more neuron types
  DenseLayer layer(std::move(neurons));
  ```

### 3. **Hybrid Activation System**
- **Per-layer or per-neuron:**  
  - Assign a single activation function to a whole layer, or
  - Assign different activation functions to each neuron.
- **Example:**
  ```cpp
  // Per-layer
  DenseLayer layer(3, 4, activations::ReLU);
  // Per-neuron
  DenseLayer layer(3, 4, {activations::ReLU, activations::Sigmoid, activations::Tanh, activations::Identity});
  ```

### 4. **Backend Abstraction (CPU/GPU)**
- **Tensor operations** can run on CPU or CUDA (if enabled).
- **Extensible** for future OpenCL/Metal support.
- **Example:**
  ```cpp
  Tensor cpu_tensor({2, 2}); // CPU
  Tensor gpu_tensor({2, 2}, Backend::CUDA); // CUDA (if enabled)
  ```

### 5. **Comprehensive Testing**
- **All neuron types** and **heterogeneous layers** are tested against hand-calculated ground truth values.
- **Test suite** covers tensor ops, neuron outputs, layer outputs, activations, and more.
- **How to run:**
  ```
  cd build
  ctest --output-on-failure
  ```

---

## Example: Heterogeneous Layer Construction

```cpp
using namespace activations;
std::vector<std::unique_ptr<Neuron>> neurons;
neurons.push_back(std::make_unique<LinearNeuron>(2, ReLU));
neurons.push_back(std::make_unique<QuadraticNeuron>(2, Identity));
neurons.push_back(std::make_unique<SirenNeuron>(2, 1.0f));
neurons.push_back(std::make_unique<RBFNeuron>(2, 1.0f));
neurons.push_back(std::make_unique<RationalNeuron>(1.0f, 1.0f));
neurons.push_back(std::make_unique<ComplexNeuron>(2, 0.0f, Identity));
DenseLayer layer(std::move(neurons));
Tensor input({2}); input({0}) = 2.0f; input({1}) = 3.0f;
float output = layer.neurons[0]->forward(input); // Forward for first neuron
```

---

## Getting Started

### Prerequisites
- C++11 or later
- CMake

### Build Instructions
```sh
git clone <repository-url>
cd deep-learning-app
mkdir build && cd build
cmake ..
make
```

### Running Tests
```sh
ctest --output-on-failure
```

---

## Design Highlights

- **Polymorphic neurons:** Add new neuron types easily.
- **Heterogeneous layers:** Mix any neuron types in a layer.
- **Hybrid activations:** Per-layer or per-neuron flexibility.
- **Backend abstraction:** CPU and CUDA support, extensible to other backends.
- **Comprehensive tests:** All core features validated against ground truth.

---

## Roadmap / Next Steps

- **Backpropagation:** Implement and test gradient computation for all neuron types.
- **Serialization:** Save/load models with all neuron and layer types.
- **More backends:** OpenCL/Metal for Mac GPU support.
- **Advanced layers:** RNNs, CNNs, attention, etc.

---

## License

MIT License. See LICENSE file for details.

---

**Contributions and feedback are welcome!**