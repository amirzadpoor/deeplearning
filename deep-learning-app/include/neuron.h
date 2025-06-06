#pragma once
#include <vector>
#include "tensor.h"
#include <functional>

class Neuron {
public:
    virtual ~Neuron() = default;
    // Forward pass: input is a 1D tensor (single sample)
    virtual float forward(const Tensor& input) = 0;
    // Backward pass: input is the same input as forward, grad_output is dL/dy, returns dL/dx
    virtual Tensor backward(const Tensor& input, float grad_output) = 0;
    // Return the input size for this neuron
    virtual int getInputSize() const = 0;
    // Update parameters using accumulated gradients
    virtual void update(float lr) = 0;
};

// Linear neuron: y = w^T x + b, then activation
class LinearNeuron : public Neuron {
public:
    LinearNeuron(int input_size, std::function<float(float)> activation = nullptr);
    float forward(const Tensor& input) override;
    Tensor backward(const Tensor& input, float grad_output) override;
    int getInputSize() const override;
    void update(float lr) override;
    void setWeights(const std::vector<float>& w);
    void setBias(float b);
    const std::vector<float>& getWeights() const { return weights; }
private:
    std::vector<float> weights;
    float bias;
    std::vector<float> grad_weights;
    float grad_bias;
    std::function<float(float)> activation;
};

// Quadratic neuron: y = x^T Q x + w^T x + b, then activation
class QuadraticNeuron : public Neuron {
public:
    QuadraticNeuron(int input_size, std::function<float(float)> activation = nullptr);
    float forward(const Tensor& input) override;
    Tensor backward(const Tensor& input, float grad_output) override;
    int getInputSize() const override;
    void update(float lr) override;
    void setQuadratic(const std::vector<std::vector<float>>& Q);
    void setWeights(const std::vector<float>& w);
    void setBias(float b);
private:
    std::vector<std::vector<float>> Q; // Quadratic term (input_size x input_size)
    std::vector<float> grad_Q;
    std::vector<float> weights;
    float bias;
    std::vector<float> grad_weights;
    float grad_bias;
    std::function<float(float)> activation;
};

// SIREN neuron: y = sin(omega * (w^T x + b))
class SirenNeuron : public Neuron {
public:
    SirenNeuron(int input_size, float omega = 30.0f);
    float forward(const Tensor& input) override;
    Tensor backward(const Tensor& input, float grad_output) override;
    int getInputSize() const override;
    void update(float lr) override;
    void setWeights(const std::vector<float>& w);
    void setBias(float b);
    void setOmega(float omega);
private:
    std::vector<float> weights;
    float bias;
    std::vector<float> grad_weights;
    float grad_bias;
    float omega;
};

// RBF neuron: y = exp(-beta * ||x - c||^2)
class RBFNeuron : public Neuron {
public:
    RBFNeuron(int input_size, float beta = 1.0f);
    float forward(const Tensor& input) override;
    Tensor backward(const Tensor& input, float grad_output) override;
    int getInputSize() const override;
    void update(float lr) override;
    void setCenter(const std::vector<float>& c);
    void setBeta(float beta);
private:
    std::vector<float> center;
    std::vector<float> grad_center;
    float beta;
    float grad_beta;
};

// Rational neuron: y = (a * x) / (1 + |b * x|)
class RationalNeuron : public Neuron {
public:
    RationalNeuron(float a = 1.0f, float b = 1.0f);
    float forward(const Tensor& input) override;
    Tensor backward(const Tensor& input, float grad_output) override;
    int getInputSize() const override;
    void update(float lr) override;
    void setA(float a);
    void setB(float b);
private:
    float a;
    float b;
    float grad_a;
    float grad_b;
};

// Complex neuron: y = f(w^T x) * exp(j * theta)
class ComplexNeuron : public Neuron {
public:
    ComplexNeuron(int input_size, float theta = 0.0f, std::function<float(float)> activation = nullptr);
    float forward(const Tensor& input) override; // Returns magnitude for now
    Tensor backward(const Tensor& input, float grad_output) override;
    int getInputSize() const override;
    void update(float lr) override;
    void setWeights(const std::vector<float>& w);
    void setTheta(float theta);
private:
    std::vector<float> weights;
    std::vector<float> grad_weights;
    float theta;
    float grad_theta;
    std::function<float(float)> activation;
};