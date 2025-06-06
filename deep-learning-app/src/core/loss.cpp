#include "loss.h"
#include <cmath>
#include <algorithm>

// Mean Squared Error Loss
float MSELoss::forward(const Tensor& prediction, const Tensor& target) const {
    const auto& pred_data = prediction.getData();
    const auto& target_data = target.getData();
    float sum = 0.0f;
    int n = pred_data.size();
    for (int i = 0; i < n; ++i) {
        float diff = pred_data[i] - target_data[i];
        sum += diff * diff;
    }
    return sum / n;
}

Tensor MSELoss::backward(const Tensor& prediction, const Tensor& target) const {
    const auto& pred_data = prediction.getData();
    const auto& target_data = target.getData();
    int n = pred_data.size();
    std::vector<float> grad_data(n);
    for (int i = 0; i < n; ++i) {
        grad_data[i] = 2.0f * (pred_data[i] - target_data[i]) / n;
    }
    return Tensor(grad_data, prediction.shape(), prediction.getBackend());
}

// L1 Loss (Mean Absolute Error)
float L1Loss::forward(const Tensor& prediction, const Tensor& target) const {
    const auto& pred_data = prediction.getData();
    const auto& target_data = target.getData();
    float sum = 0.0f;
    int n = pred_data.size();
    for (int i = 0; i < n; ++i) {
        sum += std::abs(pred_data[i] - target_data[i]);
    }
    return sum / n;
}

Tensor L1Loss::backward(const Tensor& prediction, const Tensor& target) const {
    const auto& pred_data = prediction.getData();
    const auto& target_data = target.getData();
    int n = pred_data.size();
    std::vector<float> grad_data(n);
    for (int i = 0; i < n; ++i) {
        float diff = pred_data[i] - target_data[i];
        grad_data[i] = (diff > 0 ? 1.0f : (diff < 0 ? -1.0f : 0.0f)) / n;
    }
    return Tensor(grad_data, prediction.shape(), prediction.getBackend());
}

// Cross Entropy Loss (for classification, expects probabilities)
float CrossEntropyLoss::forward(const Tensor& prediction, const Tensor& target) const {
    const auto& pred_data = prediction.getData();
    const auto& target_data = target.getData();
    float sum = 0.0f;
    int n = pred_data.size();
    for (int i = 0; i < n; ++i) {
        float p = std::max(pred_data[i], 1e-12f); // avoid log(0)
        sum -= target_data[i] * std::log(p);
    }
    return sum / n;
}

Tensor CrossEntropyLoss::backward(const Tensor& prediction, const Tensor& target) const {
    const auto& pred_data = prediction.getData();
    const auto& target_data = target.getData();
    int n = pred_data.size();
    std::vector<float> grad_data(n);
    for (int i = 0; i < n; ++i) {
        float p = std::max(pred_data[i], 1e-12f);
        grad_data[i] = -target_data[i] / p / n;
    }
    return Tensor(grad_data, prediction.shape(), prediction.getBackend());
}

// Extensibility: Users can derive from Loss to implement custom loss functions. 