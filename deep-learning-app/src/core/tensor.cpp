#include "tensor.h"
#include <iostream>
#include <vector>
#include <stdexcept>

Tensor::Tensor(const std::vector<int>& dimensions) : dimensions(dimensions) {
    int totalSize = 1;
    for (int dim : dimensions) {
        totalSize *= dim;
    }
    data.resize(totalSize);
}

Tensor::~Tensor() {
    // Destructor logic if needed
}

std::vector<int> Tensor::getDimensions() const {
    return dimensions;
}

float& Tensor::operator()(const std::vector<int>& indices) {
    int index = calculateIndex(indices);
    return data[index];
}

const float& Tensor::operator()(const std::vector<int>& indices) const {
    int index = calculateIndex(indices);
    return data[index];
}

int Tensor::calculateIndex(const std::vector<int>& indices) const {
    if (indices.size() != dimensions.size()) {
        throw std::invalid_argument("Number of indices does not match tensor dimensions.");
    }
    int index = 0;
    int stride = 1;
    for (int i = dimensions.size() - 1; i >= 0; --i) {
        index += indices[i] * stride;
        stride *= dimensions[i];
    }
    return index;
}

Tensor Tensor::operator+(const Tensor& other) const {
    if (dimensions != other.dimensions) {
        throw std::invalid_argument("Tensors must have the same dimensions for addition.");
    }
    Tensor result(dimensions);
    for (int i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (dimensions != other.dimensions) {
        throw std::invalid_argument("Tensors must have the same dimensions for multiplication.");
    }
    Tensor result(dimensions);
    for (int i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] * other.data[i];
    }
    return result;
}

Tensor Tensor::reshape(const std::vector<int>& newDimensions) {
    // Logic for reshaping the tensor
    // This is a placeholder for future implementation
    return Tensor(newDimensions);
}