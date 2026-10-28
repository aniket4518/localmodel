#pragma once

#include "tensor.h"

Tensor attention(
    const Tensor &Q,
    const Tensor &K,
    const Tensor &V);