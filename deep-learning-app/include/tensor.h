#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <iostream>

class Tensor {
public:
    Tensor(const std::vector<int>& dimensions);
    Tensor(const Tensor& other);
    ~Tensor();

    Tensor& operator=(const Tensor& other);
    Tensor operator+(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;

    void reshape(const std::vector<int>& new_dimensions);
    void print() const;

private:
    std::vector<int> dimensions;
    std::vector<float> data;
    void allocateData();
};

#endif // TENSOR_H