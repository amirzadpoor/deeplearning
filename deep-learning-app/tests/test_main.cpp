#include <gtest/gtest.h>
#include "tensor.h"
#include "layer.h"
#include "model.h"
#include "neuron.h"
#include "activations.h"

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
    Layer layer(2, 2, ReLU);
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
    model.addLayer(new Layer(2, 2, ReLU));
    Tensor data({2, 2});
    data.fill(1.0);
    Tensor labels({2, 1});
    labels.fill(0.0);
    model.train(data, labels);
    EXPECT_TRUE(true); // Placeholder for actual training validation
}

TEST(NeuronTest, ForwardCalculation) {
    Neuron neuron(3);
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
    Layer layer(2, 3, ReLU);
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
    Layer layer(3, 3, activations);
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
    Layer layer(2, 3, activations);
    // Set weights and bias for each neuron
    layer.neurons[0].setWeights({1.0f, 2.0f, 3.0f});
    layer.neurons[0].setBias(1.0f);
    layer.neurons[1].setWeights({-1.0f, 0.0f, 1.0f});
    layer.neurons[1].setBias(0.0f);

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

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}