package org.brain4j.common.lang;

import org.brain4j.common.Pair;
import org.brain4j.common.tensor.Tensor;

public interface LineSplitting {

    /**
     * Applies this function to the given arguments.
     *
     * @param line the line to split
     * @param lineIndex the index of the line in the dataset
     * @return a pair of input and output tensors
     */
    Pair<Tensor[], Tensor> apply(String line, int lineIndex);
}
