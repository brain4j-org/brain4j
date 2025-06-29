package org.brain4j.core.transformer.attention.head;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.Tensors;
import org.brain4j.core.activation.impl.SoftmaxActivation;

public class MaskedAttentionHead extends AttentionHead {

    public MaskedAttentionHead(int embedDimension, int headDimension) {
        super(embedDimension, headDimension);
    }

    @Override
    public Tensor attend(Tensor input) {
        // input = [batch_size, seq_length, embedding_dim]
        Tensor Q = input.matmulGrad(queryWeights); // [batch_size, seq_length, head_dimension]
        Tensor K = input.matmulGrad(keyWeights); // [batch_size, seq_length, head_dimension]
        Tensor V = input.matmulGrad(valueWeights); // [batch_size, seq_length, head_dimension]

        double normalizer = Math.sqrt(headDimension);

        // [seq_length, seq_length]
        Tensor scores = Q.matmulGrad(K.transpose()).div(normalizer);
        int[] shape = scores.shape();

        Tensor mask = Tensors.triangularMask(shape[shape.length - 1]);

        Tensor maskedScores = scores.addGrad(mask);
        Tensor attentionWeights = maskedScores.activateGrad(new SoftmaxActivation());

        // [seq_length, head_dimension]
        return attentionWeights.matmulGrad(V);
    }
}
