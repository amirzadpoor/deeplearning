#include "model.h"
#include "layer.h"
#include <vector>

class Model {
public:
    Model() {}

    void addLayer(Layer* layer) {
        layers.push_back(layer);
    }

    void forward(const Tensor& input) {
        Tensor output = input;
        for (auto& layer : layers) {
            output = layer->forward(output);
        }
    }

    void backward(const Tensor& outputGradient) {
        Tensor gradient = outputGradient;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            gradient = (*it)->backward(gradient);
        }
    }

    void train(const Tensor& input, const Tensor& target) {
        forward(input);
        // Compute loss and gradients here
        // Call backward with the computed output gradient
    }

private:
    std::vector<Layer*> layers;
};