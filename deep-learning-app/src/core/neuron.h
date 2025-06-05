#pragma once
#include "tensor.h"
#include <vector>

class Neuron {
public:
    Neuron(int input_size);
    float forward(const Tensor& input);
    void setWeights(const std::vector<float>& weights);
    void setBias(float bias);

private:
    std::vector<float> weights;
    float bias;
};