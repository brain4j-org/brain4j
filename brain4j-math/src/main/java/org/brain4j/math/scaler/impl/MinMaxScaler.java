package org.brain4j.math.scaler.impl;

import org.brain4j.math.Tensors;
import org.brain4j.math.scaler.FeatureScaler;
import org.brain4j.math.tensor.Tensor;

import java.util.List;

public class MinMaxScaler implements FeatureScaler {

    private final float rangeMin;
    private final float rangeMax;
    private float dataMin;
    private float dataMax;

    public MinMaxScaler(float rangeMin, float rangeMax) {
        this.rangeMin = rangeMin;
        this.rangeMax = rangeMax;
    }

    public MinMaxScaler(float rangeMin, float rangeMax, float dataMin, float dataMax) {
        this.rangeMin = rangeMin;
        this.rangeMax = rangeMax;
        this.dataMin = dataMin;
        this.dataMax = dataMax;
    }

    @Override
    public void fit(List<Tensor> tensors) {
        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;

        for (Tensor tensor : tensors) {
            float[] data = tensor.data();
            for (float v : data) {
                if (v < min) min = v;
                if (v > max) max = v;
            }
        }

        this.dataMin = min;
        this.dataMax = max;
    }

    @Override
    public Tensor transform(Tensor tensor) {
        float[] data = tensor.data();
        float[] out = new float[data.length];

        for (int i = 0; i < data.length; i++) {
            out[i] = (data[i] - dataMin) / (dataMax - dataMin)
                * (rangeMax - rangeMin) + rangeMin;
        }

        return Tensors.create(tensor.shape(), out);
    }

    public float rangeMin() {
        return rangeMin;
    }

    public float rangeMax() {
        return rangeMax;
    }

    public float dataMin() {
        return dataMin;
    }

    public float dataMax() {
        return dataMax;
    }
}