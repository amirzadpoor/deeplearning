#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"
#include <memory>
#include <string>

// Abstract base class for all loss functions
class Loss {
public:
    virtual ~Loss() = default;
    // Compute the scalar loss value
    virtual float forward(const Tensor& prediction, const Tensor& target) const = 0;
    // Compute the gradient of the loss w.r.t. prediction
    virtual Tensor backward(const Tensor& prediction, const Tensor& target) const = 0;
    // Optional: return the name of the loss
    virtual std::string name() const = 0;
};

// Mean Squared Error Loss
class MSELoss : public Loss {
public:
    float forward(const Tensor& prediction, const Tensor& target) const override;
    Tensor backward(const Tensor& prediction, const Tensor& target) const override;
    std::string name() const override { return "MSELoss"; }
};

// L1 Loss (Mean Absolute Error)
class L1Loss : public Loss {
public:
    float forward(const Tensor& prediction, const Tensor& target) const override;
    Tensor backward(const Tensor& prediction, const Tensor& target) const override;
    std::string name() const override { return "L1Loss"; }
};

// Cross Entropy Loss (for classification, expects probabilities)
class CrossEntropyLoss : public Loss {
public:
    float forward(const Tensor& prediction, const Tensor& target) const override;
    Tensor backward(const Tensor& prediction, const Tensor& target) const override;
    std::string name() const override { return "CrossEntropyLoss"; }
};

// Extensibility: Users can derive from Loss to implement custom loss functions.

#endif // LOSS_H 