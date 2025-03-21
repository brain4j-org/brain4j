package net.echo.brain4j.transformers.encoding;

import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;
import net.echo.math4j.math.vector.Vector;

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
                double angle = position / Math.pow(10000, (2.0 * i) / embeddingDim);
                double value = i % 2 == 0 ? Math.sin(angle) : Math.cos(angle);

                encodings.set(value, position, i);
            }
        }
    }

    public Vector encode(Vector input, int position) {
        Vector encoded = new Vector(input.size());

        for (int i = 0; i < input.size(); i++) {
            encoded.set(i, input.get(i) + encodings.get(position, i));
        }

        return encoded;
    }
}
