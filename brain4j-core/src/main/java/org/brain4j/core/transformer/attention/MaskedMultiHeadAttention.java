package org.brain4j.core.transformer.attention;

import org.brain4j.core.transformer.attention.head.AttentionHead;
import org.brain4j.core.transformer.attention.head.MaskedAttentionHead;

public class MaskedMultiHeadAttention extends MultiHeadAttention {

    public MaskedMultiHeadAttention(int headCount, int modelDimension) {
        super(headCount, modelDimension);
    }

    @Override
    public AttentionHead createAttentionHead() {
        return new MaskedAttentionHead(embeddingDim, headDimension);
    }
}
