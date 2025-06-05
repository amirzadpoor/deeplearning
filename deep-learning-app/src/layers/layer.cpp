#include "layer.h"
#include <stdexcept>

Layer::Layer(int num_neurons, int input_size, std::function<float(float)> activation) {
    neurons.reserve(num_neurons);
    activations.reserve(num_neurons);
    for (int i = 0; i < num_neurons; ++i) {
        neurons.emplace_back(Neuron(input_size));
        activations.push_back(activation);
    }
}

Layer::Layer(int num_neurons, int input_size, const std::vector<std::function<float(float)>>& acts) {
    if (acts.size() != num_neurons) throw std::invalid_argument("activations size must match num_neurons");
    neurons.reserve(num_neurons);
    activations = acts;
    for (int i = 0; i < num_neurons; ++i) {
        neurons.emplace_back(Neuron(input_size));
    }
}

Tensor Layer::forward(const Tensor& input) {
    // For simplicity, assume input is 1D and number of neurons matches output size
    std::vector<float> output;
    for (size_t i = 0; i < neurons.size(); ++i) {
        float z = neurons[i].forward(input);
        output.push_back(activations[i](z));
    }
    Tensor out_tensor({static_cast<int>(output.size())});
    for (size_t i = 0; i < output.size(); ++i) {
        out_tensor({static_cast<int>(i)}) = output[i];
    }
    return out_tensor;
}

Tensor Layer::backward(const Tensor& gradient) {
    // Placeholder implementation
    return gradient;
}