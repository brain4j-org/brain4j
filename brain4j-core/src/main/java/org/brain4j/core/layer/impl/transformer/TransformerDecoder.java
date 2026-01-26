package org.brain4j.core.layer.impl.transformer;

import org.brain4j.core.layer.impl.DenseLayer;
import org.brain4j.core.layer.impl.DropoutLayer;
import org.brain4j.core.layer.impl.NormLayer;
import org.brain4j.core.transformer.attention.MaskedMultiHeadAttention;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.commons.Range;

import java.util.Arrays;

/**
 * Implements a single decoder block of the Transformer architecture,
 * as introduced in the paper "Attention is All You Need".
 *
 * <p>This class extends the {@link TransformerEncoder} and reuses most of its structure.
 * Unlike the encoder, the decoder applies a causal (triangular) mask in its self-attention layer
 * to prevent attending to future positions during training and inference.
 * This implementation is better fitted for generative models (e.g. GPTs).
 *
 * <p>The expected input shape is a 3D tensor of shape {@code [batch, seq_len, embedding_dim]}.
 * The output has the same shape.
 *
 * @see TransformerEncoder
 * @see MaskedMultiHeadAttention
 * @see DenseLayer
 * @see DropoutLayer
 * @see NormLayer
 * @author xEcho1337
 */
public class TransformerDecoder extends TransformerEncoder {

    private TransformerDecoder() {
        super();
    }
    
    /**
     * Constructs a new decoder block with the specified parameters.
     * @param numHeads the amount of heads in the attention block
     * @param embeddingDim the embedding dimension of the input
     * @param dropout the dropout used when training
     */
    public TransformerDecoder(int numHeads, int embeddingDim, double dropout) {
        super(numHeads, embeddingDim, dropout);
    }

    public MultiHeadAttention createAttention(int heads, int embeddingDim) {
        return new MaskedMultiHeadAttention(clipper, heads, embeddingDim);
    }

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        checkInputLength(1, inputs);
        Tensor input = inputs[0];

        if (input.rank() != 3) {
            throw Commons.illegalArgument("Input must have shape [batch, seq_length, dimension]! Got: %s",
                Arrays.toString(input.shape()));
        }
        
        Tensor norm1 = normalizer1.forward(cache, input);
        Tensor attended = attention.forward(cache, norm1);

        if (cache.isTraining()) {
            attended = dropout.forward(cache, attended);
        }

        Tensor added = input.addGrad(attended);
        Tensor norm2 = normalizer2.forward(cache, added);

        Tensor upProjected, downProjected;
        Tensor downCache = cache.get(downProjection);

        int seqLength = input.shapeAt(1);

        if (downCache == null) {
            upProjected = upProjection.forward(cache, norm2).activateGrad(activation);
            downProjected = downProjection.forward(cache, upProjected);
        } else {
            Range[] ranges = { Range.all(), Range.point(seqLength - 1), Range.all() };
            Tensor sliced = norm2.sliceGrad(ranges);

            Tensor upProj = upProjection.forward(cache, sliced);
            Tensor activated = upProj.activateGrad(activation);
            Tensor downProj = downProjection.forward(cache, activated);
            
            downProjected = downCache.concatGrad(downProj, 1);
        }

        cache.set(downProjection, downProjected);

        if (cache.isTraining()) {
            downProjected = dropout.forward(cache, downProjected);
        }

        Tensor added2 = downProjected.addGrad(added);
        cache.rememberOutput(this, added2);

        return new Tensor[] { added2 };
    }
}
