#include "../include/tensor.h"
#include "../include/attention.h"

#include <iostream>

int main()
{
    /*
        3 tokens
        2-dimensional representation
    */

    Tensor Q({3, 2});

    Q(0, 0) = 1.0f;
    Q(0, 1) = 0.0f;

    Q(1, 0) = 0.0f;
    Q(1, 1) = 1.0f;

    Q(2, 0) = 1.0f;
    Q(2, 1) = 1.0f;


    Tensor K({3, 2});

    K(0, 0) = 1.0f;
    K(0, 1) = 0.0f;

    K(1, 0) = 0.0f;
    K(1, 1) = 1.0f;

    K(2, 0) = 1.0f;
    K(2, 1) = 1.0f;


    Tensor V({3, 2});

    V(0, 0) = 10.0f;
    V(0, 1) = 0.0f;

    V(1, 0) = 0.0f;
    V(1, 1) = 20.0f;

    V(2, 0) = 30.0f;
    V(2, 1) = 30.0f;


    Tensor output = attention(Q, K, V);

    output.print();

    return 0;
}