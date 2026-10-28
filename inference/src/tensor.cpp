#include "../include/tensor.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>

Tensor::Tensor(const std::vector<std::size_t>& shape) //constructor
    : shape_(shape)
{
    std::size_t total = 1;

    for (std::size_t dimension : shape_) {
        total *= dimension;
    }

    data_.resize(total);
}

std::size_t Tensor::numel() const  //returns total vlaue in tensor
{
    return data_.size();
}

const std::vector<std::size_t>& Tensor::shape() const  //return tensor shape using read only refrence
{
    return shape_;
}

float& Tensor::operator[](std::size_t index)//allow acces to an element using flat indexing
{
    return data_.at(index); //check where the index is within the bound
}

const float& Tensor::operator[](std::size_t index) const //read only verson used with objects
{
    return data_.at(index);
}

float& Tensor::operator()(std::size_t row, std::size_t column) //creates row and column into flat vector index
{
    std::size_t index = row * shape_[1] + column; // it allow modification of tensor(1,2),

    return data_.at(index);
}

const float& Tensor::operator()(std::size_t row, std::size_t column) const//read only version 
{
    std::size_t index = row * shape_[1] + column;

    return data_.at(index);
}

void Tensor::fill(float value) //set every element to same value
{
    std::fill(data_.begin(), data_.end(), value);
}

void Tensor::print() const
{
    std::cout << "Shape: [";

    for (std::size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i];

        if (i + 1 < shape_.size()) {
            std::cout << ", ";
        }
    }

    std::cout << "]\n";

    std::cout << "Data: ";

    for (float value : data_) {
        std::cout << value << " ";
    }

    std::cout << "\n";
}