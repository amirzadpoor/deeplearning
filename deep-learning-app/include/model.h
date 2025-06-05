#pragma once

#include <vector>
#include "layer.h"
#include "tensor.h"

class Model {
public:
    Model();
    void addLayer(Layer* layer);
    void train(const Tensor& data, const Tensor& labels);
private:
    std::vector<Layer*> layers;
};