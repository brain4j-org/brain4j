package net.echo.brain4j.transformers.attention;

import net.echo.brain4j.transformers.head.AttentionHead;
import net.echo.brain4j.transformers.head.MaskedAttentionHead;

public class MaskedMultiHeadAttention extends MultiHeadAttention {

    public MaskedMultiHeadAttention(int headCount, int modelDimension) {
        super(headCount, modelDimension);
    }

    @Override
    public AttentionHead createAttentionHead() {
        return new MaskedAttentionHead(embeddingDim, headDimension);
    }
}
