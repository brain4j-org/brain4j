package net.echo.brain4j.convolution;

import net.echo.brain4j.activation.Activation;
import net.echo.brain4j.structure.cache.Parameters;
import net.echo.math4j.math.vector.Vector;

import java.util.Random;

public class Kernel {

    private final Vector[] values;
    private final Vector[] updates;
    private final int width;
    private final int height;
    private final int id;

    public Kernel(Vector... values) {
        this.id = -1;
        this.width = values[0].size();
        this.height = values.length;
        this.values = values;
        this.updates = new Vector[height];

        for (int i = 0; i < height; i++) {
            updates[i] = new Vector(width);
        }
    }

    public Kernel(int id, int width, int height) {
        this.id = id;
        this.width = width;
        this.height = height;
        this.values = new Vector[height];
        this.updates = new Vector[height];

        for (int i = 0; i < height; i++) {
            values[i] = new Vector(width);
        }

        for (int i = 0; i < height; i++) {
            updates[i] = new Vector(width);
        }
    }

    public Kernel(int width, int height) {
        this(-1, width, height);
    }

    public static Kernel withId(int width, int height) {
        return new Kernel(Parameters.TOTAL_KERNELS++, width, height);
    }

    public int getId() {
        return id;
    }

    public void setValues(Random generator, double bound) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                double value = generator.nextDouble() * 2 * bound - bound;
                values[i].set(j, value);
            }
        }
    }

    public Kernel convolve(Kernel kernel, int padding, int stride) {
        int convWidth = (width + 2 * padding - kernel.getWidth());
        int convHeight = (height + 2 * padding - kernel.getHeight());

        if (convWidth % stride != 0 || convHeight % stride != 0) {
            throw new IllegalArgumentException("Output must be divisible by the stride! Check your kernel and filter dimensions.");
        }

        int outputWidth = convWidth / stride + 1;
        int outputHeight = convHeight / stride + 1;

        if (outputWidth <= 0 || outputHeight <= 0) {
            throw new IllegalArgumentException("Kernel dimensions must be smaller than or equal to the input dimensions");
        }

        Kernel result = new Kernel(outputWidth, outputHeight);

        for (int i = 0; i < outputHeight; i++) {
            for (int j = 0; j < outputWidth; j++) {
                double sum = 0.0;

                for (int ki = 0; ki < kernel.getHeight(); ki++) {
                    for (int kj = 0; kj < kernel.getWidth(); kj++) {
                        int y = i * stride - padding + ki;
                        int x = j * stride - padding + kj;

                        double image = 0.0;

                        if (y >= 0 && y < height && x >= 0 && x < width) {
                            image = values[y].get(x);
                        }

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

    public Kernel add(Kernel other) {
        if (width != other.getWidth() || height != other.getHeight()) {
            throw new IllegalArgumentException("Kernels must have the same size!");
        }

        Kernel result = new Kernel(width, height);

        for (int h = 0; h < height; h++) {
            Vector otherRow = other.getValues()[h];
            Vector thisRow = values[h];

            for (int w = 0; w < width; w++) {
                double otherW = otherRow.get(w);
                double thisW = thisRow.get(w);

                result.setValue(w, h, thisW + otherW);
            }
        }

        return result;
    }

    public float getValue(int x, int y) {
        return values[y].get(x);
    }

    public void setValue(int width, int height, double value) {
        values[height].set(width, value);
    }

    public Vector[] getUpdates() {
        return updates;
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

    public Kernel flip() {
        Kernel result = new Kernel(width, height);

        for (int h = 0; h < height; h++) {
            Vector row = values[h];

            for (int w = 0; w < width; w++) {
                double value = row.get(w);

                int rotatedW = width - w - 1;
                int rotatedH = height - h - 1;

                result.setValue(rotatedW, rotatedH, value);
            }
        }

        return result;
    }

    public void print() {
        for (int i = 0; i < height; i++) {
            Vector row = values[i];

            System.out.println(row.toString("%.3f"));
        }
    }

    public Kernel subtract(Kernel other) {
        if (width != other.getWidth() || height != other.getHeight()) {
            throw new IllegalArgumentException("Kernels must have the same size!");
        }

        Kernel result = new Kernel(width, height);

        for (int h = 0; h < height; h++) {
            Vector otherRow = other.getValues()[h];
            Vector thisRow = values[h];

            for (int w = 0; w < width; w++) {
                double otherW = otherRow.get(w);
                double thisW = thisRow.get(w);

                result.setValue(w, h, thisW - otherW);
            }
        }

        return result;
    }

    public void update(int w, int h, double gradient) {
        updates[h].set(w, gradient);
    }

    public void resetUpdates() {
        for (int i = 0; i < updates.length; i++) {
            updates[i] = new Vector(width);
        }
    }

    public Kernel multiply(double value) {
        Kernel result = new Kernel(width, height);

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                double multiplied = getValue(w, h) * value;
                result.setValue(w, h, multiplied);
            }
        }

        return result;
    }
}
