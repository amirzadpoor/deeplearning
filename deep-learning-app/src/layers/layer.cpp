#include "layer.h"
#include <stdexcept>
#include <memory>
#include <vector>
#include <iostream>

DenseLayer::DenseLayer(std::vector<std::unique_ptr<Neuron>> neurons)
    : neurons(std::move(neurons)) {}

DenseLayer::DenseLayer(int num_neurons, int input_size, std::function<float(float)> activation) {
    neurons.reserve(num_neurons);
    for (int i = 0; i < num_neurons; ++i) {
        neurons.push_back(std::make_unique<LinearNeuron>(input_size, activation));
    }
}

DenseLayer::DenseLayer(int num_neurons, int input_size, const std::vector<std::function<float(float)>>& activations) {
    if (activations.size() != num_neurons) throw std::invalid_argument("activations size must match num_neurons");
    neurons.reserve(num_neurons);
    for (int i = 0; i < num_neurons; ++i) {
        neurons.push_back(std::make_unique<LinearNeuron>(input_size, activations[i]));
    }
}

Tensor DenseLayer::forward(const Tensor& input) {
    auto dims = input.getDimensions();
    input_cache.clear();
    if (dims.size() == 1) {
        std::vector<float> output;
        input_cache.resize(neurons.size());
        for (size_t i = 0; i < neurons.size(); ++i) {
            output.push_back(neurons[i]->forward(input));
            input_cache[i].clear();
            input_cache[i].push_back(input);
        }
        Tensor out_tensor({static_cast<int>(output.size())}, input.getBackend());
        for (size_t i = 0; i < output.size(); ++i) {
            out_tensor({static_cast<int>(i)}) = output[i];
        }
        return out_tensor;
    } else if (dims.size() == 2) {
        int batch_size = dims[0];
        int input_size = dims[1];
        std::vector<float> batch_output;
        input_cache.resize(neurons.size());
        for (auto& v : input_cache) v.clear();
        for (int b = 0; b < batch_size; ++b) {
            Tensor row = input.getRow(b);
            for (size_t i = 0; i < neurons.size(); ++i) {
                batch_output.push_back(neurons[i]->forward(row));
                input_cache[i].push_back(row);
            }
        }
        return Tensor(batch_output, {batch_size, static_cast<int>(neurons.size())}, input.getBackend());
    } else {
        throw std::invalid_argument("DenseLayer::forward only supports 1D or 2D input tensors");
    }
}

void DenseLayer::update(float lr) {
    for (auto& neuron : neurons) {
        neuron->update(lr);
    }
}

Tensor DenseLayer::backward(const Tensor& gradient) {
    auto grad_shape = gradient.shape();
    int total_size = 1;
    for (auto s : grad_shape) total_size *= s;
    if (grad_shape.empty() || total_size == 0) {
        return Tensor({});
    }
    if (!(grad_shape.size() == 1 || grad_shape.size() == 2)) {
        return Tensor({});
    }
    int batch_size, output_size;
    bool is_batch = grad_shape.size() == 2;
    if (is_batch) {
        batch_size = grad_shape[0];
        output_size = grad_shape[1];
    } else {
        batch_size = 1;
        output_size = grad_shape[0];
    }
    int input_size = neurons[0]->getInputSize();
    std::vector<float> grad_input_data(batch_size * input_size, 0.0f);
    for (int b = 0; b < batch_size; ++b) {
        for (int n = 0; n < output_size; ++n) {
            float grad_out = is_batch ? gradient({b, n}) : gradient({n});
            Tensor input = is_batch ? input_cache[n][b] : input_cache[n][0];
            Tensor grad = neurons[n]->backward(input, grad_out);
            for (int i = 0; i < input_size; ++i) {
                grad_input_data[b * input_size + i] += grad({i});
            }
        }
    }
    if (is_batch && batch_size > 1) {
        return Tensor(grad_input_data, {batch_size, input_size}, gradient.getBackend());
    } else {
        return Tensor(grad_input_data, {input_size}, gradient.getBackend());
    }
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