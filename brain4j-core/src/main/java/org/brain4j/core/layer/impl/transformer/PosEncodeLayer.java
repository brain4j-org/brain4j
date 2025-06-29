package org.brain4j.core.layer.impl.transformer;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.Tensors;
import org.brain4j.core.layer.ForwardContext;
import org.brain4j.core.layer.Layer;

import java.util.Arrays;

public class PosEncodeLayer extends Layer {

    @Override
    public Tensor forward(ForwardContext context) {
        Tensor input = context.input();
        // [batch_size, seq_length, dimension]
        int[] shape = input.shape();

        if (shape.length != 3) {
            throw new IllegalArgumentException(
                    "Expected input shape [batch_size, seq_length, dimension], got: " + Arrays.toString(shape)
            );
        }

        int batchSize = shape[0];
        int seqLength = shape[1];
        int dimension = shape[2];

        float[] resultData = input.data();

        for (int b = 0; b < batchSize; b++) {
            int batchOffset = b * seqLength * dimension;

            for (int i = 0; i < seqLength; i++) {
                Tensor add = generate(i, dimension);

                float[] data = add.data();
                int offset = batchOffset + i * dimension;

                for (int k = 0; k < dimension; k++) {
                    resultData[offset + k] += data[k];
                }
            }
        }

        return input;
    }

    @Override
    public int size() {
        return 0;
    }

    public Tensor generate(int position, int embeddingDim) {
        Tensor token = Tensors.zeros(embeddingDim);

        for (int i = 0; i < embeddingDim; i++) {
            double exponent = (2.0 * Math.floor(i / 2.0)) / embeddingDim;

            double angle = position / Math.pow(10000, exponent);
            double value = (i % 2 == 0) ? Math.sin(angle) : Math.cos(angle);

            token.set(value, i);
        }

        return token.reshape(1, embeddingDim);
    }
}
