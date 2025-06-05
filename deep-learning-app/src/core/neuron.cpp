#include "neuron.h"
#include <stdexcept>

Neuron::Neuron(int input_size) : weights(input_size, 0.0f), bias(0.0f) {}

float Neuron::forward(const Tensor& input) {
    auto dims = input.getDimensions();
    if (dims.size() != 1 || dims[0] != weights.size()) {
        throw std::invalid_argument("Input size does not match weights size.");
    }
    float sum = 0.0f;
    for (int i = 0; i < weights.size(); ++i) {
        sum += weights[i] * input({i});
    }
    return sum + bias; // No activation for simplicity
}

void Neuron::setWeights(const std::vector<float>& w) {
    if (w.size() != weights.size()) throw std::invalid_argument("Weights size mismatch.");
    weights = w;
}

void Neuron::setBias(float b) {
    bias = b;
}