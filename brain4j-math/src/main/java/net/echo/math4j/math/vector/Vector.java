package net.echo.math4j.math.vector;

import java.util.Arrays;
import java.util.Iterator;

/**
 * Represents a mathematical vector.
 * This class is deprecated and will be completely replaced by the {@link net.echo.math4j.math.tensor.Tensor} class.
 */
@Deprecated(since = "2.6.0", forRemoval = true)
public class Vector implements Cloneable, Iterable<Double> {

    private final float[] data;

    public Vector(int size) {
        this.data = new float[size];
    }

    private Vector(float[] data) {
        this.data = data;
    }

    public static Vector of(float... data) {
        return new Vector(data);
    }

    public void set(int index, double value) {
        data[index] = (float) value;
    }

    public float get(int index) {
        return data[index];
    }

    public float[] toArray() {
        return data;
    }

    public int size() {
        return data.length;
    }

    @Override
    public String toString() {
        return Arrays.toString(data);
    }

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof Vector vector) {
            return Arrays.equals(data, vector.data);
        } else {
            return false;
        }
    }

    @Override
    public Vector clone() {
        try {
            return (Vector) super.clone();
        } catch (CloneNotSupportedException e) {
            throw new AssertionError();
        }
    }

    public String toString(String format) {
        int iMax = data.length - 1;

        if (iMax == -1) return "[]";

        StringBuilder builder = new StringBuilder();
        builder.append('[');

        for (int i = 0; ; i++) {
            builder.append(format.formatted(data[i]));

            if (i == iMax) return builder.append(']').toString();

            builder.append(", ");
        }
    }

    @Override
    public Iterator<Double> iterator() {
        double[] copy = new double[data.length];

        for (int i = 0; i < data.length; i++) {
            copy[i] = data[i];
        }

        return Arrays.stream(copy).iterator();
    }
}
