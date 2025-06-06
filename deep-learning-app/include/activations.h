#pragma once
#include <cmath>
#include <functional>
#include <vector>

namespace activations {
    struct Activation {
        std::function<float(float)> fn;
        std::function<float(float)> dfn;
        Activation(std::function<float(float)> f, std::function<float(float)> df) : fn(f), dfn(df) {}
        float operator()(float x) const { return fn(x); }
        float derivative(float x) const { return dfn(x); }
    };

    static const Activation Identity = Activation(
        [](float x) { return x; },
        [](float) { return 1.0f; }
    );
    static const Activation ReLU = Activation(
        [](float x) { return x > 0 ? x : 0; },
        [](float x) { return x > 0 ? 1.0f : 0.0f; }
    );
    static const Activation Sigmoid = Activation(
        [](float x) { return 1.0f / (1.0f + std::exp(-x)); },
        [](float x) { float s = 1.0f / (1.0f + std::exp(-x)); return s * (1.0f - s); }
    );
    static const Activation Tanh = Activation(
        [](float x) { return std::tanh(x); },
        [](float x) { float t = std::tanh(x); return 1.0f - t * t; }
    );

    // Basic activations
    inline float relu(float x) { return x > 0 ? x : 0; }
    inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
    inline float identity(float x) { return x; }
    inline float tanh(float x) { return std::tanh(x); }

    // Leaky ReLU (alpha = 0.01 by default)
    inline float leaky_relu(float x, float alpha = 0.01f) { return x > 0 ? x : alpha * x; }
    static const std::function<float(float)> LeakyReLU = [](float x) { return leaky_relu(x); };

    // ELU (alpha = 1.0 by default)
    inline float elu(float x, float alpha = 1.0f) { return x >= 0 ? x : alpha * (std::exp(x) - 1); }
    static const std::function<float(float)> ELU = [](float x) { return elu(x); };

    // Swish: x * sigmoid(x)
    inline float swish(float x) { return x * sigmoid(x); }
    static const std::function<float(float)> Swish = swish;

    // Hard Sigmoid: piecewise linear approximation
    inline float hard_sigmoid(float x) { return std::max(0.0f, std::min(1.0f, 0.2f * x + 0.5f)); }
    static const std::function<float(float)> HardSigmoid = hard_sigmoid;

    // Hard Tanh: piecewise linear approximation
    inline float hard_tanh(float x) { return std::max(-1.0f, std::min(1.0f, x)); }
    static const std::function<float(float)> HardTanh = hard_tanh;

    // GELU: Gaussian Error Linear Unit (approximation)
    inline float gelu(float x) {
        return 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / 3.14159265f) * (x + 0.044715f * std::pow(x, 3))));
    }
    static const std::function<float(float)> GELU = gelu;

    // Softmax: operates on a vector, not a scalar
    inline std::vector<float> softmax(const std::vector<float>& x) {
        std::vector<float> exp_x(x.size());
        float max_x = *std::max_element(x.begin(), x.end());
        float sum = 0.0f;
        for (size_t i = 0; i < x.size(); ++i) {
            exp_x[i] = std::exp(x[i] - max_x); // for numerical stability
            sum += exp_x[i];
        }
        for (size_t i = 0; i < x.size(); ++i) {
            exp_x[i] /= sum;
        }
        return exp_x;
    }
    // No std::function wrapper for softmax since it operates on vectors
} 