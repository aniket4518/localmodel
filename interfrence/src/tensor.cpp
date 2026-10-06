#include "../include/tensor.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>

Tensor::Tensor(const std::vector<std::size_t> &shape)
    : shape_(shape)
{
    std::size_t total = 1;

    for (std::size_t dimension : shape_)
    {
        total *= dimension;
    }

    data_.resize(total);
}

std::size_t Tensor::numel() const
{
    return data_.size();
}

const std::vector<std::size_t> &Tensor::shape() const
{
    return shape_;
}

float &Tensor::operator[](std::size_t index)
{
    return data_.at(index);
}

const float &Tensor::operator[](std::size_t index) const
{
    return data_.at(index);
}

void Tensor::fill(float value)
{
    std::fill(data_.begin(), data_.end(), value);
}

void Tensor::print() const
{
    std::cout << "Shape: [";

    for (std::size_t i = 0; i < shape_.size(); ++i)
    {
        std::cout << shape_[i];

        if (i + 1 < shape_.size())
        {
            std::cout << ", ";
        }
    }

    std::cout << "]\n";

    std::cout << "Data: ";

    for (float value : data_)
    {
        std::cout << value << " ";
    }

    std::cout << "\n";
}
