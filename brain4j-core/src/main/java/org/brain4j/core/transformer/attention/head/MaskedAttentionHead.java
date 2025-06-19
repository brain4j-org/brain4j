package org.brain4j.core.transformer.attention.head;

import org.brain4j.math.activation.impl.SoftmaxActivation;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

public class MaskedAttentionHead extends AttentionHead {

    public MaskedAttentionHead(int embedDimension, int headDimension) {
        super(embedDimension, headDimension);
    }

    @Override
    public Tensor attend(Tensor input) {
        // input = [seq_length, embedding_dim]
        Tensor Q = input.matmulGrad(queryWeights); // [seq_length, head_dimension]
        Tensor K = input.matmulGrad(keyWeights); // [seq_length, head_dimension]
        Tensor V = input.matmulGrad(valueWeights); // [seq_length, head_dimension]

        double normalizer = Math.sqrt(headDimension);

        // [seq_length, seq_length]
        Tensor scores = Q.matmulGrad(K.transpose()).div(normalizer);
        Tensor mask = Tensors.triangularMask(scores.shape()[scores.dimension() - 1]).withGrad();

        Tensor maskedScores = scores.addGrad(mask);
        Tensor attentionWeights = maskedScores.activateGrad(new SoftmaxActivation());

        // [seq_length, head_dimension]
        return attentionWeights.matmulGrad(V);
    }
}
