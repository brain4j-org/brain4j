package net.echo.brain4j.transformers;

import net.echo.brain4j.transformers.attention.MaskedMultiHeadAttention;

public class TransformerDecoder extends TransformerEncoder {

    public TransformerDecoder(int numHeads, int dimension) {
        super(numHeads, dimension);
    }

    @Override
    public MaskedMultiHeadAttention createAttention() {
        return new MaskedMultiHeadAttention(weightInit, heads, dimension);
    }
}
