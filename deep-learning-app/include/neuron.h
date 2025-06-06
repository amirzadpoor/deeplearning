#pragma once
#include <vector>
#include "tensor.h"
#include <functional>
#include <memory>
#include "activations.h"
#include <limits>

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
    virtual void update(float lr, float grad_clip_min = -std::numeric_limits<float>::infinity(), float grad_clip_max = std::numeric_limits<float>::infinity()) = 0;
};

// Linear neuron: y = w^T x + b, then activation
class LinearNeuron : public Neuron {
public:
    LinearNeuron(int input_size, activations::Activation activation = activations::Identity);
    float forward(const Tensor& input) override;
    Tensor backward(const Tensor& input, float grad_output) override;
    int getInputSize() const override;
    void update(float lr, float grad_clip_min = -std::numeric_limits<float>::infinity(), float grad_clip_max = std::numeric_limits<float>::infinity()) override;
    void setWeights(const std::vector<float>& w);
    void setBias(float b);
    float getBias() const { return bias; }
    std::vector<float> getWeights() const { return weights; }
    std::vector<float> getGradWeights() const { return grad_weights; }
    float getGradBias() const { return grad_bias; }
private:
    std::vector<float> weights;
    float bias;
    std::vector<float> grad_weights;
    float grad_bias;
    activations::Activation activation;
};

// Quadratic neuron: y = x^T Q x + w^T x + b, then activation
class QuadraticNeuron : public Neuron {
public:
    QuadraticNeuron(int input_size, activations::Activation activation = activations::Identity);
    float forward(const Tensor& input) override;
    Tensor backward(const Tensor& input, float grad_output) override;
    int getInputSize() const override;
    void update(float lr, float grad_clip_min = -std::numeric_limits<float>::infinity(), float grad_clip_max = std::numeric_limits<float>::infinity()) override;
    void setQuadratic(const std::vector<std::vector<float>>& Q);
    void setWeights(const std::vector<float>& w);
    void setBias(float b);
    float getBias() const { return bias; }
    std::vector<float> getWeights() const { return weights; }
private:
    std::vector<std::vector<float>> Q; // Quadratic term (input_size x input_size)
    std::vector<float> grad_Q;
    std::vector<float> weights;
    float bias;
    std::vector<float> grad_weights;
    float grad_bias;
    activations::Activation activation;
};

// SIREN neuron: y = sin(omega * (w^T x + b))
class SirenNeuron : public Neuron {
public:
    SirenNeuron(int input_size, float omega = 30.0f);
    float forward(const Tensor& input) override;
    Tensor backward(const Tensor& input, float grad_output) override;
    int getInputSize() const override;
    void update(float lr, float grad_clip_min = -std::numeric_limits<float>::infinity(), float grad_clip_max = std::numeric_limits<float>::infinity()) override;
    float getBias() const { return bias; }
    std::vector<float> getWeights() const { return weights; }
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
    void update(float lr, float grad_clip_min = -std::numeric_limits<float>::infinity(), float grad_clip_max = std::numeric_limits<float>::infinity()) override;
    void setCenter(const std::vector<float>& c);
    void setBeta(float beta);
    const std::vector<float>& getCenter() const { return center; }
    float getBeta() const { return beta; }
private:
    std::vector<float> center;
    std::vector<float> grad_center;
    float beta;
    float grad_beta;
};

// Rational neuron: y = (a * x) / (1 + |b * x|), for 1D input
class RationalNeuron : public Neuron {
public:
    RationalNeuron(float a = 1.0f, float b = 1.0f);
    float forward(const Tensor& input) override;
    Tensor backward(const Tensor& input, float grad_output) override;
    int getInputSize() const override;
    void update(float lr, float grad_clip_min = -std::numeric_limits<float>::infinity(), float grad_clip_max = std::numeric_limits<float>::infinity()) override;
    void setA(float a);
    void setB(float b);
    float getA() const { return a; }
    float getB() const { return b; }
private:
    float a;
    float b;
    float grad_a;
    float grad_b;
};

// Complex neuron: y = f(w^T x + b) * exp(j * theta)
class ComplexNeuron : public Neuron {
public:
    ComplexNeuron(int input_size, float theta = 0.0f, activations::Activation activation = activations::Identity);
    float forward(const Tensor& input) override; // Returns magnitude for now
    Tensor backward(const Tensor& input, float grad_output) override;
    int getInputSize() const override;
    void update(float lr, float grad_clip_min = -std::numeric_limits<float>::infinity(), float grad_clip_max = std::numeric_limits<float>::infinity()) override;
    void setWeights(const std::vector<float>& w);
    void setTheta(float theta);
    void setBias(float b);
    const std::vector<float>& getWeights() const { return weights; }
    float getTheta() const { return theta; }
    float getBias() const { return bias; }
private:
    std::vector<float> weights;
    std::vector<float> grad_weights;
    float theta;
    float grad_theta;
    float bias;
    float grad_bias;
    activations::Activation activation;
};