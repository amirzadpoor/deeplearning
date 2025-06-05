#pragma once
#include "tensor.h"
#include "neuron.h"
#include <vector>
#include <functional>
#include "activations.h"

class Layer {
public:
    // Constructor for single activation function
    // Example: Layer(3, 4, activations::ReLU)
    Layer(int num_neurons, int input_size, std::function<float(float)> activation);
    // Constructor for per-neuron activation functions
    // Example: Layer(3, 4, std::vector<std::function<float(float)>>{activations::ReLU, activations::Sigmoid, activations::Tanh})
    Layer(int num_neurons, int input_size, const std::vector<std::function<float(float)>>& activations);
    virtual ~Layer() = default;

    virtual Tensor forward(const Tensor& input);
    virtual Tensor backward(const Tensor& gradient);

    std::vector<Neuron> neurons; // Exposed for testing purposes

private:
    std::vector<std::function<float(float)>> activations;
};