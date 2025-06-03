package org.brain4j.core.transformer.attention.head;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

public class MaskedAttentionHead extends AttentionHead {

    public MaskedAttentionHead(int embedDimension, int headDimension) {
        super(embedDimension, headDimension);
    }

    @Override
    public Tensor attend(Tensor input) {
        Tensor Q = input.matmul(queryWeights);
        Tensor K = input.matmul(keyWeights);
        Tensor V = input.matmul(valueWeights);

        double normalizer = Math.sqrt(headDimension);

        Tensor scores = Q.matmul(K.transpose()).div(normalizer);
        Tensor mask = Tensors.triangularMask(scores.shape()[0]);

        Tensor maskedScores = scores.add(mask);
        Tensor attentionWeights = maskedScores.softmax();

        return attentionWeights.matmul(V);
    }
}
