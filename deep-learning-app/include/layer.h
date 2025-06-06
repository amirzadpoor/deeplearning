#pragma once
#include "tensor.h"
#include "neuron.h"
#include <vector>
#include <functional>
#include <memory>
#include "activations.h"
#include <limits>

class Layer {
public:
    virtual ~Layer() = default;
    // Forward pass for a batch: input shape [batch_size, input_size], output shape [batch_size, output_size]
    virtual Tensor forward(const Tensor& input) = 0;
    // Backward pass for a batch: gradient shape [batch_size, output_size], returns gradient w.r.t. input
    virtual Tensor backward(const Tensor& gradient) = 0;
    // Update parameters using accumulated gradients
    virtual void update(float lr, float grad_clip_min = -std::numeric_limits<float>::infinity(), float grad_clip_max = std::numeric_limits<float>::infinity()) = 0;
};

class DenseLayer : public Layer {
public:
    // Heterogeneous constructor: accepts a vector of unique_ptr<Neuron>
    DenseLayer(std::vector<std::unique_ptr<Neuron>> neurons);
    // Homogeneous constructors for backward compatibility
    DenseLayer(int num_neurons, int input_size, activations::Activation activation = activations::Identity);
    DenseLayer(int num_neurons, int input_size, const std::vector<activations::Activation>& activations);
    virtual ~DenseLayer() = default;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradient) override;
    void update(float lr, float grad_clip_min = -std::numeric_limits<float>::infinity(), float grad_clip_max = std::numeric_limits<float>::infinity()) override;

    std::vector<std::unique_ptr<Neuron>> neurons; // Now heterogeneous

private:
    std::vector<activations::Activation> activations;
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