package net.echo.brain4j.layer.impl;

import net.echo.math.activation.Activations;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.StatesCache;
import net.echo.math.tensor.Tensor;
import net.echo.math.tensor.index.Range;

import java.io.DataInputStream;
import java.io.DataOutputStream;

/**
 * Represents a normalization layer, used to normalize the inputs and improve training.
 */
public class LayerNorm extends Layer {

    private double epsilon;

    /**
     * Constructs a layer normalization instance with a default epsilon.
     */
    public LayerNorm() {
        this(1e-5);
    }

    /**
     * Constructs a layer normalization instance with an epsilon.
     * @param epsilon the epsilon used to avoid division by zero
     */
    public LayerNorm(double epsilon) {
        super(0, Activations.LINEAR);
        this.epsilon = epsilon;
    }

    @Override
    public void serialize(DataOutputStream stream) throws Exception {
        super.serialize(stream);
        stream.writeDouble(epsilon);
    }

    @Override
    public void deserialize(DataInputStream stream) throws Exception {
        super.deserialize(stream);
        this.epsilon = stream.readDouble();
    }

    @Override
    public boolean canPropagate() {
        return false;
    }

    @Override
    public Tensor forward(StatesCache cache, Layer lastLayer, Layer nextLayer, Tensor input, boolean training) {
        return normalize1D(input);
    }

    /**
     * Normalizes a tensor with a mean of zero and variance of one.
     *
     * @param input the input tensor to normalize
     * @return the normalized output tensor
     */
    public Tensor normalize(Tensor input) {
        Tensor normalized = input.clone();

        int rows = input.shape()[0];
        int columns = input.shape()[1];

        for (int i = 0; i < rows; i++) {
            Range range = new Range(i * columns, (i + 1) * columns);
            Tensor token = input.reshape(columns * rows).slice(range);

            double mean = token.mean();
            double variance = 0.0;

            for (float value : token) {
                variance += Math.pow(value - mean, 2);
            }

            variance /= token.elements();

            double denominator = Math.sqrt(variance + epsilon);

            for (int j = 0; j < token.elements(); j++) {
                double value = token.get(j);
                double normalizedValue = (value - mean) / denominator;

                normalized.set(normalizedValue, i, j);
            }
        }

        return normalized;
    }

    public Tensor normalize1D(Tensor input) {
        double mean = input.mean();
        double variance = 0.0;

        for (float value : input) {
            variance += Math.pow(value - mean, 2);
        }

        variance /= input.elements();

        double denominator = Math.sqrt(variance + epsilon);
        Tensor normalized = input.clone();

        return normalized.map(value -> (value - mean) / denominator);
    }
}
