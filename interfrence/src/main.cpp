#include "../include/tensor.h"

#include <iostream>

int main()
{
    Tensor x({2, 3});

    x[0] = 1.0f;
    x[1] = 2.0f;
    x[2] = 3.0f;

    x[3] = 4.0f;
    x[4] = 5.0f;
    x[5] = 6.0f;

    x.print();

    std::cout << "Number of elements: "
              << x.numel()
              << "\n";

    return 0;
}