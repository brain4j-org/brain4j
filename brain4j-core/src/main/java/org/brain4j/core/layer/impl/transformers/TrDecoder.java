package org.brain4j.core.layer.impl.transformers;

import org.brain4j.core.transformers.attention.MaskedMultiHeadAttention;

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
