package org.brain4j.math.tensor.autograd.operations;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;
import org.brain4j.math.tensor.autograd.Operation;
import org.brain4j.math.tensor.index.Range;

import java.util.List;

public class ConcatOperation implements Operation {

    @Override
    public Tensor forward(Tensor... inputs) {
        return Tensors.concat(List.of(inputs));
    }

    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor... inputs) {
        Tensor[] gradients = new Tensor[inputs.length];

        int dim = gradOutput.shape().length - 1;
        int start = 0;

        for (int i = 0; i < inputs.length; i++) {
            Tensor input = inputs[i];
            int size = input.shape()[dim];

            Range range = new Range(start, start + size, 1);
            Tensor gradSlice = gradOutput.slice(range);

            gradients[i] = gradSlice;
            start += size;
        }

        return gradients;
    }
}
