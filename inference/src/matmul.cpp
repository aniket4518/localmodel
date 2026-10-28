#include "../include/matmul.h"

#include <stdexcept>

Tensor matmul(const Tensor& A, const Tensor& B)
{
    const auto& shapeA = A.shape();
    const auto& shapeB = B.shape();

    if (shapeA.size() != 2 || shapeB.size() != 2) {
        throw std::runtime_error("matmul currently supports only 2D tensors");
    }

    std::size_t M = shapeA[0];
    std::size_t K = shapeA[1];

    std::size_t K2 = shapeB[0];
    std::size_t N = shapeB[1];

    if (K != K2) {
        throw std::runtime_error("Matrix dimensions do not match");
    }

    Tensor C({M, N});

    C.fill(0.0f);

    for (std::size_t i = 0; i < M; ++i) {

        for (std::size_t j = 0; j < N; ++j) {

            for (std::size_t k = 0; k < K; ++k) {

                C(i, j) += A(i, k) * B(k, j);
            }
        }
    }

    return C;
}