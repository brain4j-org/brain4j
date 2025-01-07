package net.echo.brain4j.utils;

import java.util.Arrays;
import java.util.function.Supplier;

public class Vector {

    private final double[] data;

    public Vector(int size) {
        this.data = new double[size];
    }

    private Vector(double... data) {
        this.data = data;
    }

    public static Vector of(double... data) {
        return new Vector(Arrays.copyOf(data, data.length));
    }

    public static Vector random(int size) {
        return new Vector(size).fill(Math::random);
    }

    public static Vector uniform(double lowerBound, double upperBound, int size) {
        return new Vector(size).fill(() -> Math.random() * (upperBound - lowerBound) + lowerBound);
    }

    public static Vector zero(int size) {
        return new Vector(size).fill(0.0);
    }

    public void set(int index, double value) {
        data[index] = value;
    }

    public double get(int index) {
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

    public Vector normalizeSquared() {
        double length = lengthSquared();

        for (int i = 0; i < data.length; i++) {
            data[i] /= length;
        }

        return this;
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
            throw new IllegalArgumentException("Vectors must be of the same length.");
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

    public Vector convoluted(Vector other) {
        double[] result = new double[data.length + other.data.length - 1];

        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < other.data.length; j++) {
                result[i + j] += data[i] * other.data[j];
            }
        }

        return new Vector(result);
    }

    public Vector add(Vector other) {
        for (int i = 0; i < data.length; i++) {
            data[i] += other.data[i];
        }

        return this;
    }

    public Vector scale(double value) {
        for (int i = 0; i < data.length; i++) {
            data[i] *= value;
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
        for (int i = 0; i < data.length; i++) {
            data[i] *= vector.data[i];
        }

        return this;
    }

    public double weightedSum(Vector vector) {
        double sum = 0.0;

        for (int i = 0; i < data.length; i++) {
            double value = data[i] * vector.data[i];

            sum += value;

            data[i] = value;
        }

        return sum;
    }

    public Vector fill(double value) {
        Arrays.fill(data, value);
        return this;
    }

    public Vector fill(Supplier<Double> function) {
        for (int i = 0; i < data.length; i++) {
            data[i] = function.get();
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

    public double[] toArray() {
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
}
