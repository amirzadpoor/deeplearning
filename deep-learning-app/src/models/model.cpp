#include "model.h"
#include "layer.h"
#include <vector>

Model::Model() {}

void Model::addLayer(Layer* layer) {
    layers.push_back(layer);
}

void Model::train(const Tensor& data, const Tensor& labels) {
    // Placeholder implementation
}