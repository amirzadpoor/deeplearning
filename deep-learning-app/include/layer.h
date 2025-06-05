#pragma once
#include "tensor.h"
#include "neuron.h"
#include <vector>
#include <functional>
#include "activations.h"

class Layer {
public:
    virtual ~Layer() = default;
    // Forward pass for a batch: input shape [batch_size, input_size], output shape [batch_size, output_size]
    virtual Tensor forward(const Tensor& input) = 0;
};

class DenseLayer : public Layer {
public:
    // Constructor for single activation function
    // Example: DenseLayer(3, 4, activations::ReLU)
    DenseLayer(int num_neurons, int input_size, std::function<float(float)> activation);
    // Constructor for per-neuron activation functions
    // Example: DenseLayer(3, 4, std::vector<std::function<float(float)>>{activations::ReLU, activations::Sigmoid, activations::Tanh})
    DenseLayer(int num_neurons, int input_size, const std::vector<std::function<float(float)>>& activations);
    virtual ~DenseLayer() = default;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradient);

    std::vector<Neuron> neurons; // Exposed for testing purposes

private:
    std::vector<std::function<float(float)>> activations;
};

// Example: ActivationLayer (applies an activation function to its input)
class ActivationLayer : public Layer {
public:
    ActivationLayer(std::function<float(float)> activation);
    Tensor forward(const Tensor& input) override;
private:
    std::function<float(float)> activation;
};