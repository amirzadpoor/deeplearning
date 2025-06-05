#pragma once
#include "tensor.h"

class Layer {
public:
    Layer() = default;
    virtual ~Layer() = default;

    virtual void forward(const Tensor& input);
    virtual Tensor backward(const Tensor& gradient);
};