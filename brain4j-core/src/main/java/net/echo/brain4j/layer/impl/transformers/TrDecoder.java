package net.echo.brain4j.layer.impl.transformers;

import net.echo.brain4j.transformers.attention.MaskedMultiHeadAttention;

public class TrDecoder extends TrEncoder {

    private TrDecoder() {
        super();
    }

    public TrDecoder(int numHeads, int embeddingDim) {
        super(numHeads, embeddingDim);
    }

    @Override
    public MaskedMultiHeadAttention createAttention(int heads, int embeddingDim) {
        return new MaskedMultiHeadAttention(heads, embeddingDim);
    }
}
