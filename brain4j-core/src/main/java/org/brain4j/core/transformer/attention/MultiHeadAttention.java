package org.brain4j.core.transformer.attention;

import org.brain4j.core.training.StatesCache;
import org.brain4j.core.transformer.attention.head.AttentionHead;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;
import org.brain4j.math.tensor.autograd.operations.ConcatOperation;
import org.brain4j.math.weights.WeightInitialization;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class MultiHeadAttention {

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

        initializeHeads();
    }

    public AttentionHead createAttentionHead() {
        return new AttentionHead(embeddingDim, headDimension);
    }

    public void compile(Random generator, WeightInitialization weightInit) {
        for (AttentionHead head : heads) {
            head.initWeights(generator, weightInit);
        }
    }

    public Tensor attend(StatesCache cache, Tensor input) {
        Tensor[] outputs = new Tensor[heads.size()];

        for (int i = 0; i < heads.size(); i++) {
            outputs[i] = heads.get(i).attend(cache, input);
        }

        Tensor first = outputs[0];

        int[] shape = first.shape().clone();
        shape[shape.length - 1] = 0;

        return Tensors.zeros(shape).forward(new ConcatOperation(), outputs);
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
}
