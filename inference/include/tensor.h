#pragma once

#include <vector>
#include <cstddef>

class Tensor {
private:
    std::vector<float> data_;
    std::vector<std::size_t> shape_;

public:
    Tensor(const std::vector<std::size_t>& shape);

    std::size_t numel() const;

    const std::vector<std::size_t>& shape() const;//read only version of total no of element

    float& operator[](std::size_t index);//provide access to one element of tnesor tensor[0]
    const float& operator[](std::size_t index) const;//read only verson o tensor[x]

    float& operator()(std::size_t row, std::size_t column);//proveide access to and 2d element and can modify
    const float& operator()(std::size_t row, std::size_t column) const;//read only version

    void fill(float value);

    void print() const;// print tensor shape and data
};