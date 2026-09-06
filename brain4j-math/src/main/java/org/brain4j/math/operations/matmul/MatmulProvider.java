package org.brain4j.math.operations.matmul;

import org.brain4j.math.tensor.Tensor;

public interface MatmulProvider {

    void multiply(Tensor a, Tensor b, Tensor c);
}
