#include "tensor.h"

Tensor::Tensor(const std::vector<int>& dimensions) : dimensions(dimensions) {
    // Calculate the total number of elements and allocate memory for the tensor data
    int totalElements = 1;
    for (int dim : dimensions) {
        totalElements *= dim;
    }
    data.resize(totalElements);
}

Tensor::Tensor(const Tensor& other) : dimensions(other.dimensions), data(other.data) {}

Tensor::~Tensor() {}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        dimensions = other.dimensions;
        data = other.data;
    }
    return *this;
}

Tensor Tensor::operator+(const Tensor& other) const {
    // Assuming both tensors have the same dimensions for simplicity
    Tensor result(dimensions);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    // Assuming both tensors have the same dimensions for simplicity
    Tensor result(dimensions);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] * other.data[i];
    }
    return result;
}

Tensor Tensor::reshape(const std::vector<int>& newDimensions) {
    // For simplicity, we won't actually change the data layout in memory
    return Tensor(newDimensions);
}

float& Tensor::operator()(const std::vector<int>& indices) {
    int index = calculateIndex(indices);
    return data[index];
}

const float& Tensor::operator()(const std::vector<int>& indices) const {
    int index = calculateIndex(indices);
    return data[index];
}

std::vector<int> Tensor::getDimensions() const {
    return dimensions;
}

int Tensor::calculateIndex(const std::vector<int>& indices) const {
    int index = 0;
    int multiplier = 1;
    for (size_t i = 0; i < dimensions.size(); ++i) {
        index += indices[i] * multiplier;
        multiplier *= dimensions[i];
    }
    return index;
}

Layer::Layer() {}

Layer::~Layer() {}