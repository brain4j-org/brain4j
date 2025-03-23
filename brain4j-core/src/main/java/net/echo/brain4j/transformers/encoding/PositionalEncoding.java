package net.echo.brain4j.transformers.encoding;

import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;

public class PositionalEncoding {

    private final int maxLength;
    private final int embeddingDim;
    private final Tensor encodings;

    public PositionalEncoding(int maxLength, int embeddingDim) {
        this.maxLength = maxLength;
        this.embeddingDim = embeddingDim;
        this.encodings = TensorFactory.matrix(maxLength, embeddingDim);
        initializeEncodings();
    }

    private void initializeEncodings() {
        for (int position = 0; position < maxLength; position++) {
            for (int i = 0; i < embeddingDim; i++) {
                double exponent = (2.0 * Math.floor(i / 2.0)) / embeddingDim;

                double angle = position / Math.pow(10000, exponent);
                double value = (i % 2 == 0) ? Math.sin(angle) : Math.cos(angle);
                encodings.set(value, position, i);
            }
        }
    }


    public Tensor encode(Tensor input) {
        if (input.dimension() != 2) {
            throw new IllegalArgumentException("Input must be a 2D matrix!");
        }

        int rows = input.shape()[0];
        int cols = input.shape()[1];

        Tensor encoded = TensorFactory.matrix(rows, cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                float inputValue = input.get(i, j);
                float encoding = encodings.get(i, j);

                encoded.set(inputValue + encoding, i, j);
            }
        }

        return encoded;
    }
}
