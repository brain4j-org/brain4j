package net.echo.brain4j.layer.impl.transformers;

import net.echo.brain4j.transformers.attention.MaskedMultiHeadAttention;

public class TrDecoder extends TrEncoder {

    private TrDecoder() {
        super();
    }

    public TrDecoder(int numHeads, int dimension) {
        super(numHeads, dimension);
    }

    @Override
    public MaskedMultiHeadAttention createAttention(int heads, int dimension) {
        return new MaskedMultiHeadAttention(heads, dimension);
    }
}
