package net.echo.brain4j.utils;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;

public class Vector implements Cloneable, Iterable<Double> {

    private final float[] data;

    public Vector(int size) {
        this.data = new float[size];
    }

    private Vector(float... data) {
        this.data = data;
    }

    private Vector(double... data) {
        this.data = new float[data.length];

        for (int i = 0; i < data.length; i++) {
            this.data[i] = (float) data[i];
        }
    }

    private Vector(int... data) {
        this.data = new float[data.length];

        for (int i = 0; i < data.length; i++) {
            this.data[i] = data[i];
        }
    }

    public static Vector of(float... data) {
        return new Vector(Arrays.copyOf(data, data.length));
    }

    public static Vector of(double... data) {
        return new Vector(Arrays.copyOf(data, data.length));
    }

    public static Vector of(int... data) {
        return new Vector(data);
    }

    public static Vector random(int size) {
        return new Vector(size).fill(Math::random);
    }

    public static Vector random(int seed, int size) {
        Random random = new Random(seed);

        return new Vector(size).fill(random::nextDouble);
    }

    public static Vector uniform(double lowerBound, double upperBound, int size) {
        return new Vector(size).fill(() -> Math.random() * (upperBound - lowerBound) + lowerBound);
    }

    public static Vector uniform(int seed, double lowerBound, double upperBound, int size) {
        Random random = new Random(seed);
        return new Vector(size).fill(() -> random.nextDouble() * (upperBound - lowerBound) + lowerBound);
    }

    public static Vector zeros(int size) {
        return new Vector(size).fill(0.0);
    }

    public static Vector parse(List<String> pixels) {
        Vector vector = new Vector(pixels.size());

        for (int i = 0; i < pixels.size(); i++) {
            vector.set(i, Double.parseDouble(pixels.get(i)));
        }

        return vector;
    }

    public void set(int index, double value) {
        data[index] = (float) value;
    }

    public void add(int index, double value) {
        data[index] += (float) value;
    }

    public float get(int index) {
        return data[index];
    }

    public double lengthSquared() {
        double sum = 0;

        for (double value : data) {
            sum += value * value;
        }

        return sum;
    }

    public double length() {
        return Math.sqrt(lengthSquared());
    }

    public double sum() {
        double sum = 0;

        for (double value : data) {
            sum += value;
        }

        return sum;
    }

    public double max() {
        double max = Double.NEGATIVE_INFINITY;

        for (double value : data) {
            max = Math.max(max, value);
        }

        return max;
    }

    public double min() {
        double min = Double.POSITIVE_INFINITY;

        for (double value : data) {
            min = Math.min(min, value);
        }

        return min;
    }

    public Vector normalizeSquared() {
        double length = lengthSquared();

        for (int i = 0; i < data.length; i++) {
            data[i] /= length;
        }

        return this;
    }

    public Vector normalizeMinMax() {
        double min = min();
        double max = max();

        Vector normalized = new Vector(size());

        for (int i = 0; i < size(); i++) {
            normalized.set(i, (get(i) - min) / (max - min));
        }

        return normalized;
    }

    public Vector normalize() {
        double length = length();

        for (int i = 0; i < data.length; i++) {
            data[i] /= length;
        }

        return this;
    }

    public double distanceSquared(Vector vector) {
        if (data.length != vector.data.length) {
            throw new IllegalArgumentException("Vectors must be of the same length. (" + data.length + " != " + vector.data.length + ")");
        }

        double sum = 0;

        for (int i = 0; i < data.length; i++) {
            double difference = data[i] - vector.data[i];
            sum += (difference) * (difference);
        }

        return sum;
    }

    public double distance(Vector vector) {
        return Math.sqrt(distanceSquared(vector));
    }

    public Vector add(Vector other) {
        for (int i = 0; i < data.length; i++) {
            data[i] += other.data[i];
        }

        return this;
    }

    public void subtract(Vector vector) {
        for (int i = 0; i < data.length; i++) {
            data[i] -= vector.data[i];
        }
    }

    public Vector add(double value) {
        for (int i = 0; i < data.length; i++) {
            data[i] += value;
        }

        return this;
    }

    public Vector scale(double value) {
        for (int i = 0; i < data.length; i++) {
            data[i] *= (float) value;
        }

        return this;
    }

    public Vector divide(double value) {
        for (int i = 0; i < data.length; i++) {
            data[i] /= value;
        }

        return this;
    }

    public Vector multiply(Vector vector) {
        if (data.length != vector.data.length) {
            throw new IllegalArgumentException("Vectors must be of the same length.");
        }

        for (int i = 0; i < data.length; i++) {
            data[i] *= vector.data[i];
        }

        return this;
    }

    public double weightedSum(Vector vector) {
        double sum = 0.0;

        for (int i = 0; i < data.length; i++) {
            sum += data[i] * vector.data[i];
        }

        return sum;
    }

    public Vector fill(double value) {
        Arrays.fill(data, (float) value);
        return this;
    }

    public Vector fill(Supplier<Double> function) {
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) (double) function.get();
        }

        return this;
    }

    public double mean() {
        return sum() / data.length;
    }

    public double variance(double mean) {
        double sum = 0;

        for (double datum : data) {
            sum += Math.pow(datum - mean, 2);
        }

        return sum / data.length;
    }

    public double variance() {
        double mean = mean();
        double sum = 0;

        for (double datum : data) {
            sum += Math.pow(datum - mean, 2);
        }

        return sum / data.length;
    }

    public double[] toDoubleArray() {
        double[] result = new double[data.length];

        for (int i = 0; i < data.length; i++) {
            result[i] = data[i];
        }

        return result;
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
            builder.append(String.format(format, data[i]));

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
