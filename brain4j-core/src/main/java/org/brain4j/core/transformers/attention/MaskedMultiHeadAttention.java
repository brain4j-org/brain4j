package org.brain4j.core.transformers.attention;

import org.brain4j.core.transformers.head.AttentionHead;
import org.brain4j.core.transformers.head.MaskedAttentionHead;

public class MaskedMultiHeadAttention extends MultiHeadAttention {

    public MaskedMultiHeadAttention(int headCount, int modelDimension) {
        super(headCount, modelDimension);
    }

    @Override
    public AttentionHead createAttentionHead() {
        return new MaskedAttentionHead(embeddingDim, headDimension);
    }
}
