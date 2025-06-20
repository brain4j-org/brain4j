package org.brain4j.core.transformer.attention.head;

import org.brain4j.core.training.StatesCache;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.activation.impl.SoftmaxActivation;
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

        this.queryWeights = Tensors.zeros(embedDimension, headDimension).withGrad();
        this.keyWeights = Tensors.zeros(embedDimension, headDimension).withGrad();
        this.valueWeights = Tensors.zeros(embedDimension, headDimension).withGrad();
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
        Tensor attentionWeights = scores.activateGrad(new SoftmaxActivation());

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

    public void backward(Updater updater, Optimizer optimizer) {
        Tensor queryGrad = queryWeights.grad();
        Tensor keyGrad = keyWeights.grad();
        Tensor valueGrad = valueWeights.grad();

        Tensor optimizedQuery = optimizer.step(queryWeights, queryGrad);
        Tensor optimizedKey = optimizer.step(keyWeights, keyGrad);
        Tensor optimizedValue = optimizer.step(valueWeights, valueGrad);

        updater.change(queryWeights, optimizedQuery);
        updater.change(keyWeights, optimizedKey);
        updater.change(valueWeights, optimizedValue);
    }
}
