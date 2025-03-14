package net.echo.brain4j.transformers.encoding;

import net.echo.brain4j.utils.math.vector.Vector;

public class PositionalEncoding {

    private final int maxLength;
    private final int embeddingDim;
    private final double[][] encodings;

    public PositionalEncoding(int maxLength, int embeddingDim) {
        this.maxLength = maxLength;
        this.embeddingDim = embeddingDim;
        this.encodings = new double[maxLength][embeddingDim];
        initializeEncodings();
    }

    private void initializeEncodings() {
        for (int pos = 0; pos < maxLength; pos++) {
            for (int i = 0; i < embeddingDim; i++) {
                double angle = pos / Math.pow(10000, (2.0 * i) / embeddingDim);
                encodings[pos][i] = i % 2 == 0 ? Math.sin(angle) : Math.cos(angle);
            }
        }
    }

    public Vector encode(Vector input, int position) {
        Vector encoded = new Vector(input.size());

        for (int i = 0; i < input.size(); i++) {
            encoded.set(i, input.get(i) + encodings[position][i]);
        }

        return encoded;
    }
}
