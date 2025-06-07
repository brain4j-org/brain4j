package org.brain4j.core.transformer.attention.head;

import org.brain4j.core.training.StatesCache;
import org.brain4j.math.device.DeviceType;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;
import org.brain4j.math.weights.WeightInitialization;

import java.util.Random;

public class AttentionHead {

    protected Tensor queryWeights;
    protected Tensor keyWeights;
    protected Tensor valueWeights;

    protected int embedDimension;
    protected int headDimension;

    public AttentionHead(int embedDimension, int headDimension) {
        this.embedDimension = embedDimension;
        this.headDimension = headDimension;

        this.queryWeights = Tensors.zeros(embedDimension, headDimension);
        this.keyWeights = Tensors.zeros(embedDimension, headDimension);
        this.valueWeights = Tensors.zeros(embedDimension, headDimension);
    }

    public void initWeights(Random generator, WeightInitialization weightInit) {
        this.queryWeights.map(x -> weightInit.generate(generator, embedDimension, headDimension));
        this.keyWeights.map(x -> weightInit.generate(generator, embedDimension, headDimension));
        this.valueWeights.map(x -> weightInit.generate(generator, embedDimension, headDimension));
    }

    public void to(DeviceType deviceType) {
        this.queryWeights.to(deviceType);
        this.keyWeights.to(deviceType);
        this.valueWeights.to(deviceType);
    }

    public Tensor attend(Tensor input) {
        // input = [seq_length, embedding_dim]
        Tensor Q = input.matmulGrad(queryWeights); // [seq_length, head_dimension]
        Tensor K = input.matmulGrad(keyWeights); // [seq_length, head_dimension]
        Tensor V = input.matmulGrad(valueWeights); // [seq_length, head_dimension]

        double normalizer = Math.sqrt(headDimension);

        // [seq_length, seq_length]
        Tensor scores = Q.matmulGrad(K.transpose()).div(normalizer);
        Tensor attentionWeights = scores.softmax();

        // [seq_length, head_dimension]
        return attentionWeights.matmulGrad(V);
    }

    public Tensor attend(StatesCache cache, Tensor input) {
        return attend(input);
    }

    public Tensor queryWeights() {
        return queryWeights;
    }

    public Tensor keyWeights() {
        return keyWeights;
    }

    public Tensor valueWeights() {
        return valueWeights;
    }

    public int embedDimension() {
        return embedDimension;
    }

    public int headDimension() {
        return headDimension;
    }

    public int totalWeights() {
        return queryWeights.elements() + keyWeights.elements() + valueWeights.elements();
    }
}
