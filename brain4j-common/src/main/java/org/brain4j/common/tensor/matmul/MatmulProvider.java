package org.brain4j.common.tensor.matmul;

import org.brain4j.common.tensor.Tensor;

import java.util.concurrent.ForkJoinPool;

public interface MatmulProvider {

    void multiply(ForkJoinPool pool, Tensor a, Tensor b, Tensor c);
}
