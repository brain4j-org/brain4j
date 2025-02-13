package net.echo.brain4j.convolution;

import net.echo.brain4j.activation.Activation;
import net.echo.brain4j.utils.Vector;

import java.util.Random;

public class Kernel {

    private final Vector[] values;

    private final int width;
    private final int height;

    public Kernel(Vector... values) {
        this.values = values;
        this.width = values.length;
        this.height = values[0].size();
    }

    public Kernel(int width, int height) {
        this.width = width;
        this.height = height;
        this.values = new Vector[height];

        for (int i = 0; i < height; i++) {
            values[i] = new Vector(width);
        }
    }

    public void setValues(Random generator, double bound) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                double value = generator.nextDouble() * 2 * bound - bound;
                values[i].set(j, value);
            }
        }
    }

    public Kernel convolute(Kernel kernel, int stride) {
        int outputWidth = width - kernel.getWidth() + 1;
        int outputHeight = height - kernel.getHeight() + 1;

        if (outputWidth <= 0 || outputHeight <= 0) {
            throw new IllegalArgumentException("Kernel dimensions must be smaller than or equal to the input dimensions");
        }

        Kernel result = new Kernel(outputWidth, outputHeight);

        for (int i = 0; i < outputHeight; i += stride) {
            for (int j = 0; j < outputWidth; j += stride) {
                double sum = 0.0;

                for (int ki = 0; ki < kernel.getHeight(); ki++) {
                    for (int kj = 0; kj < kernel.getWidth(); kj++) {
                        double image = values[i + ki].get(j + kj);
                        double filter = kernel.getValues()[ki].get(kj);

                        sum += image * filter;
                    }
                }

                result.getValues()[i].set(j, sum);
            }
        }

        return result;
    }

    public Kernel padding(int padding) {
        int newWidth = width + 2 * padding;
        int newHeight = height + 2 * padding;
        Kernel paddedKernel = new Kernel(newWidth, newHeight);

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                paddedKernel.getValues()[i + padding].set(j + padding, values[i].get(j));
            }
        }

        return paddedKernel;
    }

    public void add(Kernel feature) {
        for (int h = 0; h < height; h++) {
            Vector add = feature.getValues()[h];
            values[h].add(add);
        }
    }

    public double getValue(int x, int y) {
        return values[y].get(x);
    }

    public void setValue(int width, int height, double value) {
        values[height].set(width, value);
    }

    public Vector[] getValues() {
        return values;
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public int size() {
        return width * height;
    }

    public void apply(Activation activation) {
        for (int h = 0; h < height; h++) {
            Vector row = values[h];

            values[h] = activation.activate(row);
        }
    }

    public void print() {
        for (int h = 0; h < height; h++) {
            System.out.println(values[h]);
        }
    }
}
