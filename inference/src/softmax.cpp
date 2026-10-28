#include "../include/softmax.h"

#include <cmath>
#include <stdexcept>

Tensor softmax(const Tensor& input)
{
    const auto& shape = input.shape();

    if (shape.size() != 1) {
        throw std::runtime_error(
            "softmax currently supports only 1D tensors"
        );
    }

    Tensor output(shape);

    // Step 1: Find maximum value
    float max_value = input[0];

    for (std::size_t i = 1; i < input.numel(); ++i) {
        if (input[i] > max_value) {
            max_value = input[i];
        }
    }

    // Step 2: Compute exponentials
    float sum = 0.0f;

    for (std::size_t i = 0; i < input.numel(); ++i) {
        output[i] = std::exp(input[i] - max_value);
        sum += output[i];
    }

    // Step 3: Normalize
    for (std::size_t i = 0; i < input.numel(); ++i) {
        output[i] /= sum;
    }

    return output;
}