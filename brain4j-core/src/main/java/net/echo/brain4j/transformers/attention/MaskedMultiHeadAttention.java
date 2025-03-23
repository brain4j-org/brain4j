package net.echo.brain4j.transformers.attention;

import net.echo.brain4j.model.initialization.WeightInitializer;
import net.echo.brain4j.transformers.head.AttentionHead;
import net.echo.brain4j.transformers.head.MaskedAttentionHead;

public class MaskedMultiHeadAttention extends MultiHeadAttention {

    public MaskedMultiHeadAttention(WeightInitializer weightInit, int headCount, int modelDimension) {
        super(weightInit, headCount, modelDimension);
    }

    @Override
    public AttentionHead createAttentionHead() {
        return new MaskedAttentionHead(weightInit, modelDimension, headDimension);
    }
}
