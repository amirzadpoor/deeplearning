#pragma once
#include "tensor.h"
#include "neuron.h"
#include <vector>
#include <functional>
#include <memory>
#include "activations.h"

class Layer {
public:
    virtual ~Layer() = default;
    // Forward pass for a batch: input shape [batch_size, input_size], output shape [batch_size, output_size]
    virtual Tensor forward(const Tensor& input) = 0;
    // Backward pass for a batch: gradient shape [batch_size, output_size], returns gradient w.r.t. input
    virtual Tensor backward(const Tensor& gradient) = 0;
    // Update parameters using accumulated gradients
    virtual void update(float lr) = 0;
};

class DenseLayer : public Layer {
public:
    // Heterogeneous constructor: accepts a vector of unique_ptr<Neuron>
    DenseLayer(std::vector<std::unique_ptr<Neuron>> neurons);
    // Homogeneous constructors for backward compatibility
    DenseLayer(int num_neurons, int input_size, std::function<float(float)> activation);
    DenseLayer(int num_neurons, int input_size, const std::vector<std::function<float(float)>>& activations);
    virtual ~DenseLayer() = default;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradient) override;
    void update(float lr) override;

    std::vector<std::unique_ptr<Neuron>> neurons; // Now heterogeneous

private:
    std::vector<std::function<float(float)>> activations;
    std::vector<std::vector<Tensor>> input_cache;
};

// Example: ActivationLayer (applies an activation function to its input)
class ActivationLayer : public Layer {
public:
    ActivationLayer(std::function<float(float)> activation);
    Tensor forward(const Tensor& input) override;
private:
    std::function<float(float)> activation;
};