#include "layer.h"
#include <stdexcept>

DenseLayer::DenseLayer(int num_neurons, int input_size, std::function<float(float)> activation) {
    neurons.reserve(num_neurons);
    activations.reserve(num_neurons);
    for (int i = 0; i < num_neurons; ++i) {
        neurons.emplace_back(Neuron(input_size));
        activations.push_back(activation);
    }
}

DenseLayer::DenseLayer(int num_neurons, int input_size, const std::vector<std::function<float(float)>>& acts) {
    if (acts.size() != num_neurons) throw std::invalid_argument("activations size must match num_neurons");
    neurons.reserve(num_neurons);
    activations = acts;
    for (int i = 0; i < num_neurons; ++i) {
        neurons.emplace_back(Neuron(input_size));
    }
}

Tensor DenseLayer::forward(const Tensor& input) {
    auto dims = input.getDimensions();
    if (dims.size() == 1) {
        // Single sample, original behavior
        std::vector<float> output;
        for (size_t i = 0; i < neurons.size(); ++i) {
            float z = neurons[i].forward(input);
            output.push_back(activations[i](z));
        }
        Tensor out_tensor({static_cast<int>(output.size())}, input.getBackend());
        for (size_t i = 0; i < output.size(); ++i) {
            out_tensor({static_cast<int>(i)}) = output[i];
        }
        return out_tensor;
    } else if (dims.size() == 2) {
        // Batch input: shape [batch_size, input_size]
        int batch_size = dims[0];
        int input_size = dims[1];
        std::vector<float> batch_output;
        for (int b = 0; b < batch_size; ++b) {
            Tensor row = input.getRow(b);
            std::vector<float> output;
            for (size_t i = 0; i < neurons.size(); ++i) {
                float z = neurons[i].forward(row);
                output.push_back(activations[i](z));
            }
            batch_output.insert(batch_output.end(), output.begin(), output.end());
        }
        // Output shape: [batch_size, num_neurons]
        return Tensor(batch_output, {batch_size, static_cast<int>(neurons.size())}, input.getBackend());
    } else {
        throw std::invalid_argument("DenseLayer::forward only supports 1D or 2D input tensors");
    }
}

Tensor DenseLayer::backward(const Tensor& gradient) {
    // Placeholder implementation
    return gradient;
}

ActivationLayer::ActivationLayer(std::function<float(float)> activation)
    : activation(activation) {}

Tensor ActivationLayer::forward(const Tensor& input) {
    auto dims = input.getDimensions();
    std::vector<float> out_data(input.shape()[0] * (dims.size() > 1 ? input.shape()[1] : 1));
    if (dims.size() == 1) {
        for (int i = 0; i < input.shape()[0]; ++i) {
            out_data[i] = activation(input({i}));
        }
        return Tensor(out_data, {input.shape()[0]}, input.getBackend());
    } else if (dims.size() == 2) {
        int rows = dims[0];
        int cols = dims[1];
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                out_data[i * cols + j] = activation(input({i, j}));
            }
        }
        return Tensor(out_data, {rows, cols}, input.getBackend());
    } else {
        throw std::invalid_argument("ActivationLayer::forward only supports 1D or 2D input tensors");
    }
}