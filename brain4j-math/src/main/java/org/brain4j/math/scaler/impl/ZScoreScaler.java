package org.brain4j.math.scaler.impl;

import org.brain4j.math.Tensors;
import org.brain4j.math.scaler.FeatureScaler;
import org.brain4j.math.tensor.Tensor;

import java.util.List;

public class ZScoreScaler implements FeatureScaler {

    private float mean;
    private float std;

    public ZScoreScaler() {
    }

    public ZScoreScaler(float mean, float std) {
        this.mean = mean;
        this.std = std;
    }

    @Override
    public void fit(List<Tensor> tensors) {
        float sum = 0f;
        int count = 0;

        for (Tensor tensor : tensors) {
            float[] data = tensor.data();
            for (float v : data) {
                sum += v;
                count++;
            }
        }

        this.mean = sum / count;

        double varianceSum = 0f;
        for (Tensor tensor : tensors) {
            float[] data = tensor.data();
            for (float v : data) {
                varianceSum += Math.pow(v - mean, 2);
            }
        }

        this.std = (float) Math.sqrt(varianceSum / count);
    }

    @Override
    public Tensor transform(Tensor tensor) {
        float[] data = tensor.data();
        float[] out = new float[data.length];

        for (int i = 0; i < data.length; i++) {
            out[i] = (data[i] - mean) / std;
        }

        return Tensors.create(tensor.shape(), out);
    }

    public float mean() {
        return mean;
    }

    public float std() {
        return std;
    }
}
