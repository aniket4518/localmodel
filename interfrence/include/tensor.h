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

    const std::vector<std::size_t>& shape() const;

    float& operator[](std::size_t index);
    const float& operator[](std::size_t index) const;

    void fill(float value);

    void print() const;
};