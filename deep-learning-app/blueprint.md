# Blueprint: Heterogeneous, Sophisticated Neuron Types in Deep Learning Framework

## 1. Design Principles
- **Polymorphic Neurons:** Each neuron can be a different type (linear, quadratic, SIREN, RBF, etc.), with its own parameters and activation.
- **Heterogeneous Layers:** A layer can contain a mix of neuron types.
- **Per-neuron Activation:** Each neuron can have its own activation function, or use a built-in one.
- **Backward Compatibility:** Existing DenseLayer and per-layer/per-neuron activation APIs must continue to work.
- **Extensibility:** Easy to add new neuron types in the future.

---

## 2. Class Structure

### 2.1. Neuron Base Class
- Abstract base class with a virtual `forward` method.
- Optionally, a virtual `backward` for future autograd.

```cpp
class Neuron {
public:
    virtual ~Neuron() = default;
    virtual float forward(const Tensor& input) = 0;
    // Optionally: virtual void backward(...) = 0;
};
```

### 2.2. Derived Neuron Types
- Each derived class implements its own parameters and computation:
    - `LinearNeuron`
    - `QuadraticNeuron`
    - `SirenNeuron`
    - `RBFNeuron`
    - `RationalNeuron`
    - `ComplexNeuron`
    - (etc.)

Each can take an activation function (or have a built-in one).

### 2.3. Layer Class
- **HeterogeneousLayer**: Holds a vector of `std::unique_ptr<Neuron>`.
- **DenseLayer**: For backward compatibility, can be a special case where all neurons are the same type.

```cpp
class HeterogeneousLayer : public Layer {
public:
    HeterogeneousLayer(std::vector<std::unique_ptr<Neuron>> neurons);
    Tensor forward(const Tensor& input) override;
private:
    std::vector<std::unique_ptr<Neuron>> neurons;
};
```

---

## 3. Neuron Construction and Layer API

- **Factory/Builder Pattern:**  
  Provide a way to construct neurons of different types, with different activations and parameters, and assemble them into a layer.

- **User API Example:**
```cpp
std::vector<std::unique_ptr<Neuron>> neurons;
neurons.push_back(std::make_unique<QuadraticNeuron>(...));
neurons.push_back(std::make_unique<SirenNeuron>(...));
neurons.push_back(std::make_unique<LinearNeuron>(...));
auto layer = HeterogeneousLayer(std::move(neurons));
```

- **For homogeneous layers:**  
  Provide a helper to create N identical neurons easily.

---

## 4. Parameter Management
- Each neuron type manages its own parameters (weights, biases, etc.).
- Provide serialization (save/load) and initialization methods for each type.

---

## 5. Activation Functions
- Each neuron can take a `std::function<float(float)>` as an activation, or use a built-in one (e.g., SIREN uses sine).
- For rational/spline activations, parameters are part of the neuron.

---

## 6. Tensor API
- No major changes needed, but ensure that the neuron forward methods can accept the right input shapes (1D for a single sample, etc.).

---

## 7. Testing and Examples
- Add tests for each neuron type (unit and integration).
- Add examples showing how to build layers with mixed neuron types and activations.

---

## 8. Performance Considerations
- Heterogeneous layers will be less efficient than homogeneous ones (can't use matrix ops).
- For production, recommend homogeneous layers for speed, but keep heterogeneous for research.

---

## 9. Backward Compatibility
- Keep the existing DenseLayer API for users who want all neurons to be the same.
- Optionally, refactor DenseLayer to use the new infrastructure internally.

---

## 10. Documentation
- Update README and code comments to explain the new neuron and layer system.
- Provide usage examples for both homogeneous and heterogeneous layers.

---

## 11. Optional: Advanced Features
- **Low-rank/factorized quadratic neurons** for efficiency.
- **Mixed-precision support** for memory/energy savings.
- **Serialization and model export** for all neuron types.

---

## Summary Table

| Step                | Description                                      |
|---------------------|--------------------------------------------------|
| Neuron base class   | Abstract, virtual forward                        |
| Derived neurons     | Linear, Quadratic, SIREN, RBF, Rational, Complex |
| Layer class         | Holds vector of unique_ptr<Neuron>               |
| Factory/builder     | For easy construction of mixed/homogeneous layers|
| Per-neuron activation| Each neuron can have its own activation         |
| Backward compatibility| Keep DenseLayer API                            |
| Testing             | Unit/integration tests for all neuron types      |
| Documentation       | Update README, add examples                      |

---

## Next Steps

1. Refactor Neuron to be an abstract base class.
2. Implement at least two new neuron types (e.g., Quadratic, SIREN) as proof of concept.
3. Refactor DenseLayer or create HeterogeneousLayer to use `std::unique_ptr<Neuron>`.
4. Update tests and documentation. 