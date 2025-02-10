package net.echo.brain4j.convolution;

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
            this.values[i] = new Vector(width);
        }
    }

    public void setValues(Random generator, double bound) {
        for (int i = 0; i < height; i++) {
            Vector vector = values[i];

            for (int j = 0; j < width; j++) {
                vector.set(j, (generator.nextDouble() * 2 * bound) - bound);
            }
        }
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

    public Kernel convolute(Kernel kernel) {
        int outputWidth = this.width - kernel.getWidth() + 1;
        int outputHeight = this.height - kernel.getHeight() + 1;

        if (outputWidth <= 0 || outputHeight <= 0) {
            throw new IllegalArgumentException("Kernel dimensions must be smaller than or equal to the input dimensions");
        }

        Kernel result = new Kernel(outputWidth, outputHeight);

        for (int i = 0; i < outputHeight; i++) {
            for (int j = 0; j < outputWidth; j++) {
                double sum = 0.0;

                for (int ki = 0; ki < kernel.getHeight(); ki++) {
                    for (int kj = 0; kj < kernel.getWidth(); kj++) {
                        double image = this.values[i + ki].get(j + kj);
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
        int newWidth = this.width + 2 * padding;
        int newHeight = this.height + 2 * padding;
        Kernel paddedKernel = new Kernel(newWidth, newHeight);

        for (int i = 0; i < this.height; i++) {
            for (int j = 0; j < this.width; j++) {
                paddedKernel.getValues()[i + padding].set(j + padding, this.values[i].get(j));
            }
        }

        return paddedKernel;
    }

    public double getValue(int x, int y) {
        return this.values[y].get(x);
    }

    public void setValue(int width, int height, double value) {
        this.values[height].set(width, value);
    }

    public void add(Kernel feature) {
        for (int h = 0; h < this.height; h++) {
            Vector add = feature.getValues()[h];
            this.values[h].add(add);
        }
    }
}
