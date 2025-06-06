#pragma once

#include <vector>
#include <memory>
#include "layer.h"
#include "tensor.h"
#include "loss.h"
#include <limits>

class Model {
public:
    Model() = default;
    // Add a layer to the model (takes ownership)
    void addLayer(std::unique_ptr<Layer> layer);
    // Forward pass through all layers
    Tensor forward(const Tensor& input);
    // Backward pass through all layers (in reverse)
    Tensor backward(const Tensor& grad_output);
    // Update all layers' parameters
    void update(float lr, float grad_clip_min = -std::numeric_limits<float>::infinity(), float grad_clip_max = std::numeric_limits<float>::infinity());
    // Perform a single training step: forward, loss, backward, update
    float trainStep(const Tensor& input, const Tensor& target, Loss& loss, float lr, float grad_clip_min = -std::numeric_limits<float>::infinity(), float grad_clip_max = std::numeric_limits<float>::infinity());
    // Access layers (for advanced use)
    const std::vector<std::unique_ptr<Layer>>& getLayers() const { return layers; }
private:
    std::vector<std::unique_ptr<Layer>> layers;
};