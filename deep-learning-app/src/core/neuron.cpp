#include "neuron.h"
#include <stdexcept>
#include <cmath>

static float identity(float x) { return x; }

// LinearNeuron implementation
LinearNeuron::LinearNeuron(int input_size, std::function<float(float)> activation)
    : weights(input_size, 0.0f), bias(0.0f), activation(activation ? activation : identity) {}

float LinearNeuron::forward(const Tensor& input) {
    auto dims = input.getDimensions();
    if (dims.size() != 1 || dims[0] != weights.size()) {
        throw std::invalid_argument("Input size does not match weights size.");
    }
    float sum = 0.0f;
    for (int i = 0; i < weights.size(); ++i) {
        sum += weights[i] * input({i});
    }
    return activation(sum + bias);
}

void LinearNeuron::setWeights(const std::vector<float>& w) {
    if (w.size() != weights.size()) throw std::invalid_argument("Weights size mismatch.");
    weights = w;
}

void LinearNeuron::setBias(float b) {
    bias = b;
}

void LinearNeuron::update(float lr) {
    for (int i = 0; i < weights.size(); ++i) {
        weights[i] -= lr * grad_weights[i];
        grad_weights[i] = 0.0f;
    }
    bias -= lr * grad_bias;
    grad_bias = 0.0f;
}

Tensor LinearNeuron::backward(const Tensor& input, float grad_output) {
    // dy/dx = weights, dL/dx = dL/dy * dy/dx
    Tensor grad_input({static_cast<int>(weights.size())}, input.getBackend());
    if (grad_weights.size() != weights.size()) grad_weights.resize(weights.size(), 0.0f);
    for (int i = 0; i < weights.size(); ++i) {
        grad_input({i}) = grad_output * weights[i];
        grad_weights[i] += grad_output * input({i});
    }
    grad_bias += grad_output;
    return grad_input;
}

int LinearNeuron::getInputSize() const {
    return static_cast<int>(weights.size());
}

// QuadraticNeuron implementation
QuadraticNeuron::QuadraticNeuron(int input_size, std::function<float(float)> activation)
    : Q(input_size, std::vector<float>(input_size, 0.0f)), weights(input_size, 0.0f), bias(0.0f), activation(activation ? activation : identity) {}

float QuadraticNeuron::forward(const Tensor& input) {
    auto dims = input.getDimensions();
    if (dims.size() != 1 || dims[0] != weights.size()) {
        throw std::invalid_argument("Input size does not match weights size.");
    }
    // Quadratic term: x^T Q x
    float quad = 0.0f;
    for (int i = 0; i < Q.size(); ++i) {
        for (int j = 0; j < Q[i].size(); ++j) {
            quad += input({i}) * Q[i][j] * input({j});
        }
    }
    // Linear term: w^T x
    float lin = 0.0f;
    for (int i = 0; i < weights.size(); ++i) {
        lin += weights[i] * input({i});
    }
    return activation(quad + lin + bias);
}

void QuadraticNeuron::setQuadratic(const std::vector<std::vector<float>>& Q_) {
    if (Q_.size() != Q.size() || Q_[0].size() != Q[0].size()) throw std::invalid_argument("Quadratic matrix size mismatch.");
    Q = Q_;
}

void QuadraticNeuron::setWeights(const std::vector<float>& w) {
    if (w.size() != weights.size()) throw std::invalid_argument("Weights size mismatch.");
    weights = w;
}

void QuadraticNeuron::setBias(float b) {
    bias = b;
}

void QuadraticNeuron::update(float lr) {
    for (int i = 0; i < weights.size(); ++i) {
        weights[i] -= lr * grad_weights[i];
        grad_weights[i] = 0.0f;
    }
    bias -= lr * grad_bias;
    grad_bias = 0.0f;
}

Tensor QuadraticNeuron::backward(const Tensor& input, float grad_output) {
    // dy/dx = 2*Q*x + w
    int n = static_cast<int>(weights.size());
    Tensor grad_input({n}, input.getBackend());
    if (grad_weights.size() != weights.size()) grad_weights.resize(weights.size(), 0.0f);
    for (int i = 0; i < n; ++i) {
        float qterm = 0.0f;
        for (int j = 0; j < n; ++j) {
            qterm += Q[i][j] * input({j});
        }
        grad_input({i}) = grad_output * (2.0f * qterm + weights[i]);
        grad_weights[i] += grad_output * (2.0f * qterm + weights[i]);
    }
    grad_bias += grad_output;
    return grad_input;
}

int QuadraticNeuron::getInputSize() const {
    return static_cast<int>(weights.size());
}

// SIREN neuron: y = sin(omega * (w^T x + b))
SirenNeuron::SirenNeuron(int input_size, float omega)
    : weights(input_size, 0.0f), bias(0.0f), omega(omega) {}

float SirenNeuron::forward(const Tensor& input) {
    auto dims = input.getDimensions();
    if (dims.size() != 1 || dims[0] != weights.size()) {
        throw std::invalid_argument("Input size does not match weights size.");
    }
    float sum = 0.0f;
    for (int i = 0; i < weights.size(); ++i) {
        sum += weights[i] * input({i});
    }
    return std::sin(omega * (sum + bias));
}

void SirenNeuron::setWeights(const std::vector<float>& w) {
    if (w.size() != weights.size()) throw std::invalid_argument("Weights size mismatch.");
    weights = w;
}

void SirenNeuron::setBias(float b) { bias = b; }

void SirenNeuron::setOmega(float o) { omega = o; }

void SirenNeuron::update(float lr) {
    for (int i = 0; i < weights.size(); ++i) {
        weights[i] -= lr * grad_weights[i];
        grad_weights[i] = 0.0f;
    }
    bias -= lr * grad_bias;
    grad_bias = 0.0f;
}

Tensor SirenNeuron::backward(const Tensor& input, float grad_output) {
    // y = sin(omega * (w^T x + b))
    // dy/dx_i = cos(omega * (w^T x + b)) * omega * w_i
    float sum = 0.0f;
    for (int i = 0; i < weights.size(); ++i) sum += weights[i] * input({i});
    float cos_term = std::cos(omega * (sum + bias));
    Tensor grad_input({static_cast<int>(weights.size())}, input.getBackend());
    if (grad_weights.size() != weights.size()) grad_weights.resize(weights.size(), 0.0f);
    for (int i = 0; i < weights.size(); ++i) {
        grad_input({i}) = grad_output * cos_term * omega * weights[i];
        grad_weights[i] += grad_output * cos_term * omega * weights[i];
    }
    grad_bias += grad_output * cos_term * omega;
    return grad_input;
}

int SirenNeuron::getInputSize() const {
    return static_cast<int>(weights.size());
}

// RBF neuron: y = exp(-beta * ||x - c||^2)
RBFNeuron::RBFNeuron(int input_size, float beta)
    : center(input_size, 0.0f), beta(beta) {}

float RBFNeuron::forward(const Tensor& input) {
    auto dims = input.getDimensions();
    if (dims.size() != 1 || dims[0] != center.size()) {
        throw std::invalid_argument("Input size does not match center size.");
    }
    float dist2 = 0.0f;
    for (int i = 0; i < center.size(); ++i) {
        float d = input({i}) - center[i];
        dist2 += d * d;
    }
    return std::exp(-beta * dist2);
}

void RBFNeuron::setCenter(const std::vector<float>& c) {
    if (c.size() != center.size()) throw std::invalid_argument("Center size mismatch.");
    center = c;
}

void RBFNeuron::setBeta(float b) { beta = b; }

void RBFNeuron::update(float lr) {
    if (grad_center.size() != center.size()) grad_center.resize(center.size(), 0.0f);
    for (int i = 0; i < center.size(); ++i) {
        center[i] -= lr * grad_center[i];
        grad_center[i] = 0.0f;
    }
    beta -= lr * grad_beta;
    grad_beta = 0.0f;
}

Tensor RBFNeuron::backward(const Tensor& input, float grad_output) {
    float y = forward(input);
    Tensor grad_input({static_cast<int>(center.size())}, input.getBackend());
    if (grad_center.size() != center.size()) grad_center.resize(center.size(), 0.0f);
    for (int i = 0; i < center.size(); ++i) {
        float grad = grad_output * (-2.0f * beta * (input({i}) - center[i]) * y);
        grad_input({i}) = grad;
        grad_center[i] += grad;
    }
    // Gradient for beta
    float dist2 = 0.0f;
    for (int i = 0; i < center.size(); ++i) {
        float d = input({i}) - center[i];
        dist2 += d * d;
    }
    grad_beta += grad_output * (-dist2 * y);
    return grad_input;
}

int RBFNeuron::getInputSize() const {
    return static_cast<int>(center.size());
}

// Rational neuron: y = (a * x) / (1 + |b * x|), for 1D input
RationalNeuron::RationalNeuron(float a, float b) : a(a), b(b) {}

float RationalNeuron::forward(const Tensor& input) {
    auto dims = input.getDimensions();
    if (dims.size() != 1 || dims[0] != 1) {
        throw std::invalid_argument("RationalNeuron expects 1D input of size 1.");
    }
    float x = input({0});
    return (a * x) / (1.0f + std::abs(b * x));
}

void RationalNeuron::setA(float a_) { a = a_; }

void RationalNeuron::setB(float b_) { b = b_; }

void RationalNeuron::update(float lr) {
    a -= lr * grad_a;
    b -= lr * grad_b;
    grad_a = 0.0f;
    grad_b = 0.0f;
}

Tensor RationalNeuron::backward(const Tensor& input, float grad_output) {
    float x = input({0});
    float denom = 1.0f + std::abs(b * x);
    float signbx = (b * x) >= 0 ? 1.0f : -1.0f;
    float num = a * denom - a * x * b * signbx;
    float dy_dx = num / (denom * denom);
    Tensor grad_input({1}, input.getBackend());
    grad_input({0}) = grad_output * dy_dx;
    // Parameter gradients
    grad_a += grad_output * x / denom;
    grad_b += grad_output * (-a * x * signbx) / (denom * denom);
    return grad_input;
}

int RationalNeuron::getInputSize() const {
    return 1;
}

// Complex neuron: y = f(w^T x) * exp(j * theta), returns magnitude for now
ComplexNeuron::ComplexNeuron(int input_size, float theta, std::function<float(float)> activation)
    : weights(input_size, 0.0f), theta(theta), activation(activation ? activation : identity) {}

float ComplexNeuron::forward(const Tensor& input) {
    auto dims = input.getDimensions();
    if (dims.size() != 1 || dims[0] != weights.size()) {
        throw std::invalid_argument("Input size does not match weights size.");
    }
    float sum = 0.0f;
    for (int i = 0; i < weights.size(); ++i) {
        sum += weights[i] * input({i});
    }
    float mag = activation(sum);
    // Return magnitude only; phase is theta
    return mag; // Optionally: return std::polar(mag, theta) for complex support
}

void ComplexNeuron::setWeights(const std::vector<float>& w) {
    if (w.size() != weights.size()) throw std::invalid_argument("Weights size mismatch.");
    weights = w;
}

void ComplexNeuron::setTheta(float t) { theta = t; }

void ComplexNeuron::update(float lr) {
    for (int i = 0; i < weights.size(); ++i) {
        weights[i] -= lr * grad_weights[i];
        grad_weights[i] = 0.0f;
    }
    theta -= lr * grad_theta;
    grad_theta = 0.0f;
}

Tensor ComplexNeuron::backward(const Tensor& input, float grad_output) {
    // y = activation(w^T x) * exp(j*theta), return grad for magnitude only
    // dy/dx_i = activation'(w^T x) * w_i (for Identity activation, derivative is 1)
    float sum = 0.0f;
    for (int i = 0; i < weights.size(); ++i) sum += weights[i] * input({i});
    float dact = 1.0f; // Only correct for Identity activation
    Tensor grad_input({static_cast<int>(weights.size())}, input.getBackend());
    if (grad_weights.size() != weights.size()) grad_weights.resize(weights.size(), 0.0f);
    for (int i = 0; i < weights.size(); ++i) {
        grad_input({i}) = grad_output * dact * weights[i];
        grad_weights[i] += grad_output * dact * weights[i];
    }
    return grad_input;
}

int ComplexNeuron::getInputSize() const {
    return static_cast<int>(weights.size());
}

// Implementation of derived neuron types will go here.