package net.echo.brain4j.convolution;

import net.echo.brain4j.utils.Vector;

import java.util.Random;

public class Kernel {

    private final Vector[] values;

    private final int width;
    private final int height;

    public Kernel(Random generator, double bound, int width, int height) {
        this.width = width;
        this.height = height;
        this.values = new Vector[height];

        for (int i = 0; i < height; i++) {
            Vector vector = new Vector(width);

            for (int j = 0; j < width; j++) {
                vector.set(j, (generator.nextDouble() * 2 * bound) - bound);
            }

            this.values[i] = vector;
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
}
