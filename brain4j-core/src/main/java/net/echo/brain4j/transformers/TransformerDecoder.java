package net.echo.brain4j.transformers;

import net.echo.brain4j.transformers.attention.MaskedMultiHeadAttention;

public class TransformerDecoder extends TransformerEncoder {

    private TransformerDecoder() {
        super();
    }

    public TransformerDecoder(int numHeads, int dimension) {
        super(numHeads, dimension);
    }

    @Override
    public MaskedMultiHeadAttention createAttention(int heads, int dimension) {
        return new MaskedMultiHeadAttention(weightInit, heads, dimension);
    }
}
