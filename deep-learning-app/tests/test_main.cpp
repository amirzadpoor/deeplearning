#include <gtest/gtest.h>
#include "tensor.h"
#include "layer.h"
#include "model.h"
#include "neuron.h"
#include "activations.h"
#include <memory>
#include <algorithm>
#include <iostream>

using namespace activations;

TEST(TensorTest, Addition) {
    Tensor a({2, 2});
    Tensor b({2, 2});
    a.fill(1.0);
    b.fill(2.0);
    Tensor result = a + b;
    EXPECT_EQ(result.get(0, 0), 3.0);
    EXPECT_EQ(result.get(1, 1), 3.0);
}

TEST(TensorTest, Multiplication) {
    Tensor a({2, 2});
    Tensor b({2, 2});
    a.fill(2.0);
    b.fill(3.0);
    Tensor result = a * b;
    EXPECT_EQ(result.get(0, 0), 6.0);
    EXPECT_EQ(result.get(1, 1), 6.0);
}

TEST(LayerTest, ForwardBackward) {
    DenseLayer layer(2, 2, ReLU);
    Tensor input({2});
    input({0}) = 1.0f;
    input({1}) = 2.0f;
    Tensor output = layer.forward(input);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_FLOAT_EQ(output({0}), 0.0f);
    EXPECT_FLOAT_EQ(output({1}), 0.0f);
    
    Tensor grad_output({2});
    grad_output({0}) = 0.5f;
    grad_output({1}) = 0.5f;
    Tensor grad_input = layer.backward(grad_output);
    EXPECT_EQ(grad_input.shape()[0], 2);
}

TEST(ModelTest, Train) {
    Model model;
    model.addLayer(new DenseLayer(2, 2, ReLU));
    Tensor data({2, 2});
    data.fill(1.0);
    Tensor labels({2, 1});
    labels.fill(0.0);
    model.train(data, labels);
    EXPECT_TRUE(true); // Placeholder for actual training validation
}

TEST(NeuronTest, ForwardCalculation) {
    LinearNeuron neuron(3);
    neuron.setWeights({1.0f, 2.0f, 3.0f});
    neuron.setBias(0.5f);

    Tensor input({3});
    input.fill(0.0f);
    input({0}) = 1.0f;
    input({1}) = 2.0f;
    input({2}) = 3.0f;

    float output = neuron.forward(input);
    // Ground truth: 1*1 + 2*2 + 3*3 + 0.5 = 1 + 4 + 9 + 0.5 = 14.5
    EXPECT_FLOAT_EQ(output, 14.5f);
}

TEST(LayerTest, SingleActivationFunction) {
    DenseLayer layer(2, 3, ReLU);
    Tensor input({3});
    input({0}) = -1.0f;
    input({1}) = 2.0f;
    input({2}) = 3.0f;
    Tensor output = layer.forward(input);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_FLOAT_EQ(output({0}), 0.0f);
    EXPECT_FLOAT_EQ(output({1}), 0.0f);
}

TEST(LayerTest, PerNeuronActivationFunctions) {
    std::vector<std::function<float(float)>> activations = {Identity, ReLU, Sigmoid};
    DenseLayer layer(3, 3, activations);
    Tensor input({3});
    input({0}) = 1.0f;
    input({1}) = -2.0f;
    input({2}) = 0.5f;
    Tensor output = layer.forward(input);
    EXPECT_EQ(output.shape()[0], 3);
    EXPECT_FLOAT_EQ(output({0}), 0.0f);
    EXPECT_FLOAT_EQ(output({1}), 0.0f);
    EXPECT_NEAR(output({2}), 0.5f, 1e-5);
}

TEST(LayerTest, OutputMatchesGroundTruth) {
    // Two neurons, three inputs each, different activations
    std::vector<std::function<float(float)>> activations = {ReLU, Sigmoid};
    DenseLayer layer(2, 3, activations);
    // Set weights and bias for each neuron
    dynamic_cast<LinearNeuron*>(layer.neurons[0].get())->setWeights({1.0f, 2.0f, 3.0f});
    dynamic_cast<LinearNeuron*>(layer.neurons[0].get())->setBias(1.0f);
    dynamic_cast<LinearNeuron*>(layer.neurons[1].get())->setWeights({-1.0f, 0.0f, 1.0f});
    dynamic_cast<LinearNeuron*>(layer.neurons[1].get())->setBias(0.0f);

    Tensor input({3});
    input({0}) = 1.0f;
    input({1}) = 2.0f;
    input({2}) = 3.0f;

    Tensor output = layer.forward(input);

    // Hand-calculated:
    // Neuron 0: z = 1*1 + 2*2 + 3*3 + 1 = 1 + 4 + 9 + 1 = 15, ReLU(15) = 15
    // Neuron 1: z = -1*1 + 0*2 + 1*3 + 0 = -1 + 0 + 3 = 2, Sigmoid(2) ≈ 0.880797
    EXPECT_FLOAT_EQ(output({0}), 15.0f);
    EXPECT_NEAR(output({1}), 0.880797f, 1e-5);
}

TEST(ActivationsTest, ScalarActivationsGroundTruth) {
    using namespace activations;
    // ReLU
    EXPECT_FLOAT_EQ(ReLU(3.0f), 3.0f);
    EXPECT_FLOAT_EQ(ReLU(-2.0f), 0.0f);
    // Leaky ReLU (default alpha=0.01)
    EXPECT_FLOAT_EQ(LeakyReLU(3.0f), 3.0f);
    EXPECT_FLOAT_EQ(LeakyReLU(-2.0f), -0.02f);
    // ELU (default alpha=1.0)
    EXPECT_FLOAT_EQ(ELU(2.0f), 2.0f);
    EXPECT_NEAR(ELU(-1.0f), std::exp(-1.0f) - 1.0f, 1e-5);
    // Swish
    EXPECT_NEAR(Swish(2.0f), 2.0f * Sigmoid(2.0f), 1e-5);
    EXPECT_NEAR(Swish(-1.0f), -1.0f * Sigmoid(-1.0f), 1e-5);
    // Hard Sigmoid
    EXPECT_FLOAT_EQ(HardSigmoid(2.0f), 0.9f);
    EXPECT_FLOAT_EQ(HardSigmoid(-3.0f), 0.0f);
    EXPECT_FLOAT_EQ(HardSigmoid(5.0f), 1.0f);
    // Hard Tanh
    EXPECT_FLOAT_EQ(HardTanh(2.0f), 1.0f);
    EXPECT_FLOAT_EQ(HardTanh(-3.0f), -1.0f);
    EXPECT_FLOAT_EQ(HardTanh(0.5f), 0.5f);
    // GELU (approximate)
    EXPECT_NEAR(GELU(1.0f), 0.5f * 1.0f * (1.0f + std::tanh(std::sqrt(2.0f / 3.14159265f) * (1.0f + 0.044715f * std::pow(1.0f, 3)))), 1e-5);
    // Tanh
    EXPECT_FLOAT_EQ(Tanh(0.0f), 0.0f);
    EXPECT_NEAR(Tanh(1.0f), std::tanh(1.0f), 1e-5);
    // Sigmoid
    EXPECT_FLOAT_EQ(Sigmoid(0.0f), 0.5f);
    EXPECT_NEAR(Sigmoid(2.0f), 1.0f / (1.0f + std::exp(-2.0f)), 1e-5);
    // Identity
    EXPECT_FLOAT_EQ(Identity(5.0f), 5.0f);
    EXPECT_FLOAT_EQ(Identity(-3.0f), -3.0f);
}

TEST(ActivationsTest, SoftmaxGroundTruth) {
    using namespace activations;
    std::vector<float> input = {1.0f, 2.0f, 3.0f};
    std::vector<float> result = softmax(input);
    float sum = std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f);
    EXPECT_NEAR(result[0], std::exp(1.0f) / sum, 1e-5);
    EXPECT_NEAR(result[1], std::exp(2.0f) / sum, 1e-5);
    EXPECT_NEAR(result[2], std::exp(3.0f) / sum, 1e-5);
}

TEST(LayerTest, BatchForwardGroundTruth) {
    using namespace activations;
    // 2 samples, 3 inputs each, 2 neurons
    DenseLayer layer(2, 3, std::vector<std::function<float(float)>>{ReLU, Sigmoid});
    dynamic_cast<LinearNeuron*>(layer.neurons[0].get())->setWeights({1.0f, 2.0f, 3.0f});
    dynamic_cast<LinearNeuron*>(layer.neurons[0].get())->setBias(1.0f);
    dynamic_cast<LinearNeuron*>(layer.neurons[1].get())->setWeights({-1.0f, 0.0f, 1.0f});
    dynamic_cast<LinearNeuron*>(layer.neurons[1].get())->setBias(0.0f);
    // Batch input: 2 samples
    std::vector<float> flat_data = {
        1.0f, 2.0f, 3.0f, // sample 0
        0.0f, 1.0f, 0.0f  // sample 1
    };
    Tensor input(flat_data, {2, 3}, Backend::CPU);
    Tensor output = layer.forward(input);
    // Output should be shape [2, 2]
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 2);
    // Sample 0, neuron 0: z = 1+4+9+1=15, ReLU(15)=15
    // Sample 0, neuron 1: z = -1+0+3=2, Sigmoid(2)=0.880797
    // Sample 1, neuron 0: z = 0+2+0+1=3, ReLU(3)=3
    // Sample 1, neuron 1: z = 0+0+0=0, Sigmoid(0)=0.5
    EXPECT_FLOAT_EQ(output({0, 0}), 15.0f);
    EXPECT_NEAR(output({0, 1}), 0.880797f, 1e-5);
    EXPECT_FLOAT_EQ(output({1, 0}), 3.0f);
    EXPECT_FLOAT_EQ(output({1, 1}), 0.5f);
}

TEST(NeuronTest, LinearNeuronGroundTruth) {
    LinearNeuron n(3);
    n.setWeights({1.0f, 2.0f, 3.0f});
    n.setBias(0.5f);
    Tensor input({3});
    input({0}) = 1.0f; input({1}) = 2.0f; input({2}) = 3.0f;
    // y = 1*1 + 2*2 + 3*3 + 0.5 = 1 + 4 + 9 + 0.5 = 14.5
    EXPECT_FLOAT_EQ(n.forward(input), 14.5f);
}

TEST(NeuronTest, QuadraticNeuronGroundTruth) {
    QuadraticNeuron n(2);
    n.setQuadratic({{1.0f, 0.0f}, {0.0f, 2.0f}}); // Q = diag(1,2)
    n.setWeights({1.0f, 1.0f});
    n.setBias(0.0f);
    Tensor input({2});
    input({0}) = 2.0f; input({1}) = 3.0f;
    // y = [2 3] * [[1 0][0 2]] * [2;3] + [1 1]*[2;3] = (2*1*2 + 3*2*3) + (2+3) = (4+18)+5=27
    float quad = 2*1*2 + 3*2*3; // 4 + 18 = 22
    float lin = 2 + 3; // 5
    float y = quad + lin;
    EXPECT_FLOAT_EQ(n.forward(input), y);
}

TEST(NeuronTest, SirenNeuronGroundTruth) {
    SirenNeuron n(2, 2.0f);
    n.setWeights({1.0f, 2.0f});
    n.setBias(0.5f);
    Tensor input({2});
    input({0}) = 1.0f; input({1}) = 2.0f;
    // y = sin(2 * (1*1 + 2*2 + 0.5)) = sin(2 * (1 + 4 + 0.5)) = sin(2*5.5) = sin(11)
    float sum = 1 + 4 + 0.5f;
    float y = std::sin(2.0f * sum);
    EXPECT_NEAR(n.forward(input), y, 1e-5);
}

TEST(NeuronTest, RBFNeuronGroundTruth) {
    RBFNeuron n(2, 0.5f);
    n.setCenter({1.0f, 2.0f});
    Tensor input({2});
    input({0}) = 2.0f; input({1}) = 4.0f;
    // y = exp(-0.5 * ((2-1)^2 + (4-2)^2)) = exp(-0.5 * (1 + 4)) = exp(-2.5)
    float dist2 = 1 + 4;
    float y = std::exp(-0.5f * dist2);
    EXPECT_NEAR(n.forward(input), y, 1e-5);
}

TEST(NeuronTest, RationalNeuronGroundTruth) {
    RationalNeuron n(2.0f, 3.0f);
    Tensor input({1});
    input({0}) = 4.0f;
    // y = (2*4)/(1+|3*4|) = 8/13 = 0.6153846
    float y = 8.0f / 13.0f;
    EXPECT_NEAR(n.forward(input), y, 1e-5);
}

TEST(NeuronTest, ComplexNeuronGroundTruth) {
    ComplexNeuron n(2, 0.0f); // theta=0, identity activation
    n.setWeights({2.0f, 3.0f});
    Tensor input({2});
    input({0}) = 1.0f; input({1}) = 2.0f;
    // y = 2*1 + 3*2 = 2 + 6 = 8
    float y = 8.0f;
    EXPECT_FLOAT_EQ(n.forward(input), y);
}

TEST(DenseLayerTest, HeterogeneousLayerGroundTruth) {
    using namespace activations;
    std::vector<std::unique_ptr<Neuron>> neurons;
    // LinearNeuron: weights={1,2}, bias=1
    auto lin = std::make_unique<LinearNeuron>(2, ReLU);
    lin->setWeights({1.0f, 2.0f}); lin->setBias(1.0f);
    // QuadraticNeuron: Q=I, weights={0,0}, bias=0
    auto quad = std::make_unique<QuadraticNeuron>(2, Identity);
    quad->setQuadratic({{1.0f, 0.0f}, {0.0f, 1.0f}}); quad->setWeights({0.0f, 0.0f}); quad->setBias(0.0f);
    // SirenNeuron: weights={1,1}, bias=0, omega=1
    auto siren = std::make_unique<SirenNeuron>(2, 1.0f);
    siren->setWeights({1.0f, 1.0f}); siren->setBias(0.0f);
    // RBFNeuron: center={0,0}, beta=1
    auto rbf = std::make_unique<RBFNeuron>(2, 1.0f);
    rbf->setCenter({0.0f, 0.0f});
    // RationalNeuron: a=1, b=1
    auto rat = std::make_unique<RationalNeuron>(1.0f, 1.0f);
    // ComplexNeuron: weights={1,1}, theta=0
    auto comp = std::make_unique<ComplexNeuron>(2, 0.0f, Identity);
    comp->setWeights({1.0f, 1.0f});
    // Build layer
    std::vector<std::unique_ptr<Neuron>> layer_neurons;
    layer_neurons.push_back(std::move(lin));
    layer_neurons.push_back(std::move(quad));
    layer_neurons.push_back(std::move(siren));
    layer_neurons.push_back(std::move(rbf));
    layer_neurons.push_back(std::move(rat));
    layer_neurons.push_back(std::move(comp));
    DenseLayer layer(std::move(layer_neurons));
    // Input: [2, 3] (for 2D neurons), [4] for rational
    Tensor input2({2}); input2({0}) = 2.0f; input2({1}) = 3.0f;
    Tensor input1({1}); input1({0}) = 4.0f;
    // Forward pass for each neuron
    float rational_expected = (1.0f * 4.0f) / (1.0f + std::abs(1.0f * 4.0f)); // 4/5 = 0.8
    std::vector<float> expected = {
        std::max(0.0f, float(1*2+2*3+1)), // LinearNeuron: ReLU(2+6+1)=9
        2*2+3*3,                   // QuadraticNeuron: 2^2+3^2=4+9=13
        std::sin(1.0f*(2+3)),      // SirenNeuron: sin(5)
        std::exp(-1.0f*(2*2+3*3)),// RBFNeuron: exp(-1*(4+9))=exp(-13)
        rational_expected,         // RationalNeuron: 4/5=0.8
        2+3                        // ComplexNeuron: 2+3=5
    };
    // Build a batch input for the layer: [input2, input2, input2, input2, input1, input2]
    std::vector<Tensor> inputs = {input2, input2, input2, input2, input1, input2};
    std::vector<float> outputs;
    for (size_t i = 0; i < layer.neurons.size(); ++i) {
        outputs.push_back(layer.neurons[i]->forward(inputs[i]));
    }
    EXPECT_FLOAT_EQ(outputs[0], expected[0]);
    EXPECT_FLOAT_EQ(outputs[1], expected[1]);
    EXPECT_NEAR(outputs[2], expected[2], 1e-5);
    EXPECT_NEAR(outputs[3], expected[3], 1e-5);
    EXPECT_NEAR(outputs[4], expected[4], 1e-5);
    EXPECT_FLOAT_EQ(outputs[5], expected[5]);
}

TEST(NeuronTest, LinearNeuronBackwardGroundTruth) {
    LinearNeuron n(2);
    n.setWeights({2.0f, 3.0f});
    n.setBias(1.0f);
    Tensor input({2});
    input({0}) = 4.0f; input({1}) = 5.0f;
    // y = 2*4 + 3*5 + 1 = 8 + 15 + 1 = 24
    // dy/dx0 = 2, dy/dx1 = 3
    float grad_output = 1.0f; // dL/dy = 1
    Tensor grad_input = n.backward(input, grad_output);
    EXPECT_FLOAT_EQ(grad_input({0}), 2.0f);
    EXPECT_FLOAT_EQ(grad_input({1}), 3.0f);
}

TEST(NeuronTest, QuadraticNeuronBackwardGroundTruth) {
    QuadraticNeuron n(2);
    n.setQuadratic({{1.0f, 0.0f}, {0.0f, 2.0f}}); // Q = diag(1,2)
    n.setWeights({1.0f, 1.0f});
    n.setBias(0.0f);
    Tensor input({2});
    input({0}) = 2.0f; input({1}) = 3.0f;
    // y = x^T Q x + w^T x + b = 2^2*1 + 3^2*2 + 1*2 + 1*3 = 4 + 18 + 2 + 3 = 27
    // dy/dx0 = 2*Q00*x0 + w0 = 2*1*2 + 1 = 5
    // dy/dx1 = 2*Q11*x1 + w1 = 2*2*3 + 1 = 13
    float grad_output = 1.0f;
    Tensor grad_input = n.backward(input, grad_output);
    EXPECT_FLOAT_EQ(grad_input({0}), 5.0f);
    EXPECT_FLOAT_EQ(grad_input({1}), 13.0f);
}

TEST(NeuronTest, SirenNeuronBackwardGroundTruth) {
    SirenNeuron n(2, 2.0f);
    n.setWeights({1.0f, 2.0f});
    n.setBias(0.5f);
    Tensor input({2});
    input({0}) = 1.0f; input({1}) = 2.0f;
    // y = sin(2 * (1*1 + 2*2 + 0.5)) = sin(11)
    // dy/dx0 = cos(11) * 2 * 1 = 2*cos(11)
    // dy/dx1 = cos(11) * 2 * 2 = 4*cos(11)
    float sum = 1 + 4 + 0.5f;
    float grad_output = 1.0f;
    float cos11 = std::cos(2.0f * sum);
    Tensor grad_input = n.backward(input, grad_output);
    EXPECT_NEAR(grad_input({0}), 2.0f * cos11, 1e-5);
    EXPECT_NEAR(grad_input({1}), 4.0f * cos11, 1e-5);
}

TEST(NeuronTest, RBFNeuronBackwardGroundTruth) {
    RBFNeuron n(2, 0.5f);
    n.setCenter({1.0f, 2.0f});
    Tensor input({2});
    input({0}) = 2.0f; input({1}) = 4.0f;
    // y = exp(-0.5 * ((2-1)^2 + (4-2)^2)) = exp(-2.5)
    // dy/dx0 = -0.5 * 2 * (2-1) * y = -1 * 1 * y = -y
    // dy/dx1 = -0.5 * 2 * (4-2) * y = -1 * 2 * y = -2y
    float y = std::exp(-0.5f * (1 + 4));
    float grad_output = 1.0f;
    Tensor grad_input = n.backward(input, grad_output);
    EXPECT_NEAR(grad_input({0}), -y, 1e-5);
    EXPECT_NEAR(grad_input({1}), -2*y, 1e-5);
}

TEST(NeuronTest, RationalNeuronBackwardGroundTruth) {
    RationalNeuron n(2.0f, 3.0f);
    Tensor input({1});
    input({0}) = 4.0f;
    // y = (2*4)/(1+|3*4|) = 8/13
    // dy/dx = (2*(1+12) - 8*3*sign(12)) / (1+12)^2 = (2*13 - 24) / 169 = (26-24)/169 = 2/169
    float grad_output = 1.0f;
    Tensor grad_input = n.backward(input, grad_output);
    EXPECT_NEAR(grad_input({0}), 2.0f/169.0f, 1e-5);
}

TEST(NeuronTest, ComplexNeuronBackwardGroundTruth) {
    ComplexNeuron n(2, 0.0f, Identity);
    n.setWeights({2.0f, 3.0f});
    Tensor input({2});
    input({0}) = 1.0f; input({1}) = 2.0f;
    // y = 2*1 + 3*2 = 8
    // dy/dx0 = 2, dy/dx1 = 3
    float grad_output = 1.0f;
    Tensor grad_input = n.backward(input, grad_output);
    EXPECT_FLOAT_EQ(grad_input({0}), 2.0f);
    EXPECT_FLOAT_EQ(grad_input({1}), 3.0f);
}

TEST(DenseLayerTest, HeterogeneousLayerBackwardGroundTruth) {
    using namespace activations;
    std::vector<std::unique_ptr<Neuron>> neurons;
    auto lin = std::make_unique<LinearNeuron>(2, ReLU);
    lin->setWeights({1.0f, 2.0f}); lin->setBias(1.0f);
    auto quad = std::make_unique<QuadraticNeuron>(2, Identity);
    quad->setQuadratic({{1.0f, 0.0f}, {0.0f, 1.0f}}); quad->setWeights({0.0f, 0.0f}); quad->setBias(0.0f);
    auto siren = std::make_unique<SirenNeuron>(2, 1.0f);
    siren->setWeights({1.0f, 1.0f}); siren->setBias(0.0f);
    auto rbf = std::make_unique<RBFNeuron>(2, 1.0f);
    rbf->setCenter({0.0f, 0.0f});
    auto rat = std::make_unique<RationalNeuron>(1.0f, 1.0f);
    auto comp = std::make_unique<ComplexNeuron>(2, 0.0f, Identity);
    comp->setWeights({1.0f, 1.0f});
    std::vector<std::unique_ptr<Neuron>> layer_neurons;
    layer_neurons.push_back(std::move(lin));
    layer_neurons.push_back(std::move(quad));
    layer_neurons.push_back(std::move(siren));
    layer_neurons.push_back(std::move(rbf));
    layer_neurons.push_back(std::move(rat));
    layer_neurons.push_back(std::move(comp));
    DenseLayer layer(std::move(layer_neurons));
    Tensor input2({2}); input2({0}) = 2.0f; input2({1}) = 3.0f;
    Tensor input1({1}); input1({0}) = 4.0f;
    std::vector<Tensor> inputs = {input2, input2, input2, input2, input1, input2};
    std::vector<float> grad_outputs = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<Tensor> grads;
    for (size_t i = 0; i < layer.neurons.size(); ++i) {
        grads.push_back(layer.neurons[i]->backward(inputs[i], grad_outputs[i]));
    }
    // LinearNeuron: dy/dx = [1,2]
    EXPECT_FLOAT_EQ(grads[0]({0}), 1.0f);
    EXPECT_FLOAT_EQ(grads[0]({1}), 2.0f);
    // QuadraticNeuron: dy/dx = [2*2+0, 2*3+0] = [4,6]
    EXPECT_FLOAT_EQ(grads[1]({0}), 4.0f);
    EXPECT_FLOAT_EQ(grads[1]({1}), 6.0f);
    // SirenNeuron: dy/dx = cos(5)*1, cos(5)*1
    float cos5 = std::cos(1.0f*(2+3));
    EXPECT_NEAR(grads[2]({0}), cos5, 1e-5);
    EXPECT_NEAR(grads[2]({1}), cos5, 1e-5);
    // RBFNeuron: dy/dx = -2*x*y = -2*2*exp(-13), -2*3*exp(-13)
    float y = std::exp(-1.0f*(2*2+3*3));
    EXPECT_NEAR(grads[3]({0}), -2*2*y, 1e-5);
    EXPECT_NEAR(grads[3]({1}), -2*3*y, 1e-5);
    // RationalNeuron: dy/dx = (1*(1+4) - 4*1*1) / (1+4)^2 = (5-4)/25 = 0.04
    EXPECT_NEAR(grads[4]({0}), 0.04f, 1e-5);
    // ComplexNeuron: dy/dx = [1,1]
    EXPECT_FLOAT_EQ(grads[5]({0}), 1.0f);
    EXPECT_FLOAT_EQ(grads[5]({1}), 1.0f);
}

TEST(MultiLayerTest, TwoLayerForwardBackwardUpdate) {
    // Layer 1: 2 inputs -> 2 outputs, Layer 2: 2 inputs -> 1 output
    DenseLayer l1(2, 2, Identity);
    DenseLayer l2(1, 2, Identity);
    // Set weights and biases for easy calculation
    dynamic_cast<LinearNeuron*>(l1.neurons[0].get())->setWeights({1.0f, 2.0f});
    dynamic_cast<LinearNeuron*>(l1.neurons[0].get())->setBias(0.0f);
    dynamic_cast<LinearNeuron*>(l1.neurons[1].get())->setWeights({-1.0f, 1.0f});
    dynamic_cast<LinearNeuron*>(l1.neurons[1].get())->setBias(0.0f);
    dynamic_cast<LinearNeuron*>(l2.neurons[0].get())->setWeights({1.0f, 1.0f});
    dynamic_cast<LinearNeuron*>(l2.neurons[0].get())->setBias(0.0f);
    // Input
    Tensor input({2}); input({0}) = 1.0f; input({1}) = 2.0f;
    // Forward pass
    Tensor h = l1.forward(input); // h0 = 1*1+2*2=5, h1=-1*1+1*2=1
    EXPECT_FLOAT_EQ(h({0}), 5.0f);
    EXPECT_FLOAT_EQ(h({1}), 1.0f);
    Tensor out = l2.forward(h); // y = 1*5+1*1=6
    EXPECT_FLOAT_EQ(out({0}), 6.0f);
    // Backward pass: dL/dy = 1
    Tensor grad_out({1, 1}); grad_out({0, 0}) = 1.0f;
    Tensor grad_h = l2.backward(grad_out); // dL/dh0 = 1*1=1, dL/dh1=1*1=1
    Tensor grad_in = l1.backward(grad_h); // dL/dx0 = 1*1 + 1*(-1) = 0, dL/dx1 = 1*2 + 1*1 = 3
    EXPECT_FLOAT_EQ(grad_in({0}), 0.0f);
    EXPECT_FLOAT_EQ(grad_in({1}), 3.0f);
    // Parameter update (SGD, lr=0.1)
    l2.update(0.1f);
    l1.update(0.1f);
    // Check updated weights for l2
    auto l2n = dynamic_cast<LinearNeuron*>(l2.neurons[0].get());
    EXPECT_NEAR(l2n->getWeights()[0], 1.0f - 0.1f * 5.0f, 1e-5); // grad_w0 = dL/dy * h0 = 1*5
    EXPECT_NEAR(l2n->getWeights()[1], 1.0f - 0.1f * 1.0f, 1e-5); // grad_w1 = dL/dy * h1 = 1*1
    // Check updated weights for l1
    auto l1n0 = dynamic_cast<LinearNeuron*>(l1.neurons[0].get());
    auto l1n1 = dynamic_cast<LinearNeuron*>(l1.neurons[1].get());
    EXPECT_NEAR(l1n0->getWeights()[0], 1.0f - 0.1f * 1.0f, 1e-5); // grad = dL/dh0 * x0 = 1*1
    EXPECT_NEAR(l1n0->getWeights()[1], 2.0f - 0.1f * 2.0f, 1e-5); // grad = dL/dh0 * x1 = 1*2
    EXPECT_NEAR(l1n1->getWeights()[0], -1.0f - 0.1f * 1.0f, 1e-5); // grad = dL/dh1 * x0 = 1*1
    EXPECT_NEAR(l1n1->getWeights()[1], 1.0f - 0.1f * 2.0f, 1e-5); // grad = dL/dh1 * x1 = 1*2
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}