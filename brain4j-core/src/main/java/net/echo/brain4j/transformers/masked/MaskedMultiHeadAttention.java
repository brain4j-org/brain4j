package net.echo.brain4j.transformers.masked;

import net.echo.brain4j.model.initialization.WeightInitializer;
import net.echo.brain4j.transformers.attention.AttentionHead;
import net.echo.brain4j.transformers.attention.MultiHeadAttention;

public class MaskedMultiHeadAttention extends MultiHeadAttention {

    public MaskedMultiHeadAttention(WeightInitializer weightInit, int headCount, int modelDimension, double temperature) {
        super(weightInit, headCount, modelDimension, temperature);
    }

    @Override
    public AttentionHead createAttentionHead() {
        return new MaskedAttentionHead(weightInit, modelDimension, headDimension, temperature);
    }
}
