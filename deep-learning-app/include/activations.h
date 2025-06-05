#pragma once
#include <cmath>
#include <functional>

namespace activations {
    inline float relu(float x) { return x > 0 ? x : 0; }
    inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
    inline float identity(float x) { return x; }
    inline float tanh(float x) { return std::tanh(x); }

    // For convenience, provide std::function wrappers
    static const std::function<float(float)> ReLU = relu;
    static const std::function<float(float)> Sigmoid = sigmoid;
    static const std::function<float(float)> Identity = identity;
    static const std::function<float(float)> Tanh = tanh;
} 