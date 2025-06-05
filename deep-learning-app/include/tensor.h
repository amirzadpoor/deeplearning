#ifndef TENSOR_H
#define TENSOR_H

#include <vector>

class Tensor {
public:
    Tensor(const std::vector<int>& dimensions);
    Tensor(const Tensor& other);
    ~Tensor();

    Tensor& operator=(const Tensor& other);
    Tensor operator+(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor reshape(const std::vector<int>& newDimensions);

    float& operator()(const std::vector<int>& indices);
    const float& operator()(const std::vector<int>& indices) const;
    std::vector<int> getDimensions() const;

    // Batch support
    // Returns a 1D Tensor representing the i-th row of a 2D tensor
    Tensor getRow(int i) const;
    // Construct a Tensor from flat data and shape
    Tensor(const std::vector<float>& flat_data, const std::vector<int>& shape);

    void fill(float value);
    float get(int i, int j) const;
    std::vector<int> shape() const;

private:
    std::vector<int> dimensions;
    std::vector<float> data;
    int calculateIndex(const std::vector<int>& indices) const;
};

#endif // TENSOR_H