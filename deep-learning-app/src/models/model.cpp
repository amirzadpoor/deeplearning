#include "model.h"
#include "layer.h"
#include <vector>
#include <algorithm>

void Model::addLayer(std::unique_ptr<Layer> layer) {
    layers.push_back(std::move(layer));
}

Tensor Model::forward(const Tensor& input) {
    Tensor x = input;
    for (auto& layer : layers) {
        x = layer->forward(x);
    }
    return x;
}

Tensor Model::backward(const Tensor& grad_output) {
    Tensor grad = grad_output;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        grad = (*it)->backward(grad);
    }
    return grad;
}

void Model::update(float lr, float grad_clip_min, float grad_clip_max) {
    for (auto& layer : layers) {
        layer->update(lr, grad_clip_min, grad_clip_max);
    }
}

float Model::trainStep(const Tensor& input, const Tensor& target, Loss& loss, float lr, float grad_clip_min, float grad_clip_max) {
    // Forward pass
    Tensor output = forward(input);
    
    // Compute loss
    float loss_value = loss.forward(output, target);
    
    // Backward pass
    Tensor grad_output = loss.backward(output, target);
    backward(grad_output);
    
    // Update parameters
    update(lr, grad_clip_min, grad_clip_max);
    
    return loss_value;
}