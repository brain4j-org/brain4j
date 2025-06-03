package org.brain4j.core.layer.impl.transformer;

import org.brain4j.core.transformer.attention.MaskedMultiHeadAttention;
import org.brain4j.core.transformer.attention.MultiHeadAttention;

public class TransformerDecoder extends TransformerEncoder {

    public TransformerDecoder(int numHeads, int embeddingDim, double dropout) {
        super(numHeads, embeddingDim, dropout);
    }

    public MultiHeadAttention createAttention(int heads, int embeddingDim) {
        return new MaskedMultiHeadAttention(heads, embeddingDim);
    }
}
