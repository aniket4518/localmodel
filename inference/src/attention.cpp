#include "../include/attention.h"
#include "../include/matmul.h"
#include "../include/softmax.h"

#include <cmath>
#include <stdexcept>

Tensor attention(
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V
)
{
    const auto& q_shape = Q.shape();
    const auto& k_shape = K.shape();
    const auto& v_shape = V.shape();

    if (q_shape.size() != 2 ||
        k_shape.size() != 2 ||
        v_shape.size() != 2) {

        throw std::runtime_error(
            "Attention currently supports only 2D tensors"
        );
    }

    std::size_t sequence_length = q_shape[0];
    std::size_t head_dimension = q_shape[1];

    if (k_shape[0] != sequence_length ||
        k_shape[1] != head_dimension) {

        throw std::runtime_error(
            "Q and K dimensions do not match"
        );
    }

    if (v_shape[0] != sequence_length) {

        throw std::runtime_error(
            "K and V sequence lengths do not match"
        );
    }

    // Step 1:
    // Q × Kᵀ

    Tensor K_transposed({
        k_shape[1],
        k_shape[0]
    });

    for (std::size_t i = 0; i < k_shape[0]; ++i) {
        for (std::size_t j = 0; j < k_shape[1]; ++j) {

            K_transposed(j, i) = K(i, j);
        }
    }

    Tensor scores = matmul(Q, K_transposed);

    // Step 2:
    // Divide by √d

    float scale = std::sqrt(
        static_cast<float>(head_dimension)
    );

    for (std::size_t i = 0; i < scores.numel(); ++i) {
        scores[i] /= scale;
    }

    // Step 3:
    // Softmax each row
    //
    // We will implement this properly
    // for 2D tensors below.

    Tensor weights(scores.shape());

    for (std::size_t row = 0;
         row < sequence_length;
         ++row) {

        float max_value = scores(row, 0);

        for (std::size_t col = 1;
             col < sequence_length;
             ++col) {

            if (scores(row, col) > max_value) {
                max_value = scores(row, col);
            }
        }

        float sum = 0.0f;

        for (std::size_t col = 0;
             col < sequence_length;
             ++col) {

            weights(row, col) =
                std::exp(scores(row, col) - max_value);

            sum += weights(row, col);
        }

        for (std::size_t col = 0;
             col < sequence_length;
             ++col) {

            weights(row, col) /= sum;
        }
    }

    // Step 4:
    // Attention weights × V

    Tensor output = matmul(weights, V);

    return output;
}