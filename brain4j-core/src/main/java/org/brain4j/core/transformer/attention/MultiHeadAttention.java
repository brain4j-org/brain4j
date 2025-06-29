package org.brain4j.core.transformer.attention;

import org.brain4j.common.device.Device;
import org.brain4j.core.training.StatesCache;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.core.transformer.attention.head.AttentionHead;
import org.brain4j.common.device.DeviceType;
import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.Tensors;
import org.brain4j.common.weightsinit.WeightInitialization;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class MultiHeadAttention {

    protected final Tensor outProjWeights;
    protected final List<AttentionHead> heads;
    protected final int headCount;
    protected final int embeddingDim;
    protected final int headDimension;

    public MultiHeadAttention(int headCount, int embeddingDim) {
        this.headCount = headCount;
        this.embeddingDim = embeddingDim;

        if (embeddingDim % headCount != 0) {
            throw new IllegalArgumentException("Embedding dimension must be divisible by head count! (%s %% %s = %s)"
                    .formatted(embeddingDim, headCount, embeddingDim % headCount));
        }

        this.headDimension = embeddingDim / headCount;
        this.heads = new ArrayList<>();
        this.outProjWeights = Tensors.matrix(embeddingDim, embeddingDim);

        initializeHeads();
    }

    public void to(Device device) {
        for (AttentionHead head : heads) {
            head.to(device);
        }
    }

    public AttentionHead createAttentionHead() {
        return new AttentionHead(embeddingDim, headDimension);
    }

    public void compile(Random generator, WeightInitialization weightInit) {
        for (AttentionHead head : heads) {
            head.initWeights(generator, weightInit);
        }

        this.outProjWeights.map(x -> weightInit.generate(generator, embeddingDim, embeddingDim));
    }

    public Tensor attend(StatesCache cache, Tensor input) {
        Tensor[] outputs = new Tensor[heads.size()];

        for (int i = 0; i < heads.size(); i++) {
            outputs[i] = heads.get(i).attend(cache, input);
        }

        Tensor result = outputs[0];

        for (int i = 1; i < outputs.length; i++) {
            result = result.concatGrad(outputs[i]);
        }

        return result.matmulGrad(outProjWeights);
    }

    protected void initializeHeads() {
        for (int i = 0; i < headCount; i++) {
            heads.add(createAttentionHead());
        }
    }

    public int totalWeights() {
        return heads.stream().mapToInt(AttentionHead::totalWeights).sum();
    }

    public List<AttentionHead> heads() {
        return heads;
    }

    public void backward(Updater updater, Optimizer optimizer) {
        for (AttentionHead head : heads) {
            head.backward(updater, optimizer);
        }
    }
}
