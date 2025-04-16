package net.echo.brain4j.transformers;

import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;

import java.util.ArrayList;
import java.util.List;

public class PositionalEncoding {

    private final List<Tensor> encodings;
    private final int embeddingDim;

    public PositionalEncoding(int embeddingDim, int maxLength) {
        this.embeddingDim = embeddingDim;
        this.encodings = new ArrayList<>();

        initializeEncodings(maxLength);
    }

    public PositionalEncoding(int embeddingDim) {
        this(embeddingDim, 1024);
    }

    private void initializeEncodings(int maxLength) {
        for (int position = 0; position < maxLength; position++) {
            encodings.add(generate(position));
        }
    }

    public Tensor generate(int position) {
        Tensor token = TensorFactory.zeros(embeddingDim);

        for (int i = 0; i < embeddingDim; i++) {
            double exponent = (2.0 * Math.floor(i / 2.0)) / embeddingDim;

            double angle = position / Math.pow(10000, exponent);
            double value = (i % 2 == 0) ? Math.sin(angle) : Math.cos(angle);

            token.set(value, i);
        }

        return token;
    }

    public Tensor getEncoding(int index) {
        while (index >= encodings.size()) {
            encodings.add(generate(encodings.size()));
        }

        Tensor token = generate(index);
        encodings.add(token);

        return token;
    }

    public Tensor encode(Tensor input) {
        if (input.dimension() != 2) {
            throw new IllegalArgumentException("Input must be a 2D matrix!");
        }

        int rows = input.shape()[0];
        List<Tensor> tokens = TensorFactory.toList(input);

        for (int i = 0; i < rows; i++) {
            Tensor encoding = getEncoding(i);
            Tensor current = tokens.get(i);

            current.add(encoding);
        }

        return TensorFactory.mergeTensors(tokens);
    }
}
