package net.echo.brain4j.transformers.encoding;

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

    public double[] encode(double[] input, int position) {
        double[] encoded = new double[input.length];

        for (int i = 0; i < input.length; i++) {
            encoded[i] = input[i] + encodings[position][i];
        }

        return encoded;
    }
}
