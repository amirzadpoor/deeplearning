#include "tensor.h"
#include <iostream>
#include <vector>
#include <stdexcept>
#ifdef USE_CUDA
#include <cuda_runtime.h>
// Stub for a CUDA kernel launcher
void launchAddKernel(const float* a, const float* b, float* out, int size) {
    // In a real implementation, launch a CUDA kernel here
    // For now, just a placeholder
}
#endif

Tensor::Tensor(const std::vector<int>& dimensions, Backend backend)
    : dimensions(dimensions), data(), backend(backend) {
    int totalSize = 1;
    for (int dim : dimensions) {
        totalSize *= dim;
    }
    data.resize(totalSize);
}

Tensor::Tensor(const Tensor& other) : dimensions(other.dimensions), data(other.data) {}

Tensor::~Tensor() {
    // Destructor logic if needed
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        dimensions = other.dimensions;
        data = other.data;
    }
    return *this;
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
    if (backend == Backend::CUDA) {
#ifdef USE_CUDA
        // Call CUDA kernel (stub)
        Tensor result(dimensions, Backend::CUDA);
        launchAddKernel(data.data(), other.data.data(), result.data.data(), data.size());
        return result;
#else
        throw std::runtime_error("CUDA backend not available. Rebuild with USE_CUDA.");
#endif
    } else {
        // CPU implementation
        Tensor result(dimensions, Backend::CPU);
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }
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

void Tensor::fill(float value) {
    std::fill(data.begin(), data.end(), value);
}

float Tensor::get(int i, int j) const {
    if (dimensions.size() != 2) throw std::invalid_argument("get(i, j) only supports 2D tensors");
    int index = i * dimensions[1] + j;
    return data[index];
}

std::vector<int> Tensor::shape() const {
    return dimensions;
}

Tensor::Tensor(const std::vector<float>& flat_data, const std::vector<int>& shape, Backend backend)
    : dimensions(shape), data(flat_data), backend(backend) {}

Tensor Tensor::getRow(int i) const {
    // Only for 2D tensors
    if (dimensions.size() != 2) throw std::invalid_argument("getRow only supports 2D tensors");
    int cols = dimensions[1];
    std::vector<float> row_data(cols);
    for (int j = 0; j < cols; ++j) {
        row_data[j] = data[i * cols + j];
    }
    return Tensor(row_data, {cols}, backend);
}

void Tensor::toBackend(Backend new_backend) {
    // Stub: in the future, move/copy data to the new backend (CPU <-> CUDA)
    backend = new_backend;
}