package org.brain4j.core.layer.impl.transformer;

import org.brain4j.core.layer.impl.DenseLayer;
import org.brain4j.core.layer.impl.DropoutLayer;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.commons.Range;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Tensor;

import java.util.Arrays;

public class TransformerDecoder extends TransformerEncoder {
    
    public TransformerDecoder(int numHeads, int embeddingDim, double dropout) {
        super(numHeads, embeddingDim, dropout);
    }
    
    @Override
    protected MultiHeadAttention createAttention(int heads, int embeddingDim) {
        return new MaskedMultiHeadAttention(clipper, heads, embeddingDim);
    }
    
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor input = inputs[0];
        
        if (input.rank() != 3) {
            throw Commons.illegalArgument("Input must have shape [batch, seq_length, dimension]! Got: %s",
                Arrays.toString(input.shape()));
        }
        
        Tensor norm1 = normalizer1.forward(cache, input)[0];
        Tensor attended = attention.forward(cache, norm1)[0];
        
        if (cache.isTraining()) {
            attended = dropout.forward(cache, attended)[0];
        }
        
        Tensor added = input.addGrad(attended);
        Tensor norm2 = normalizer2.forward(cache, added)[0];
        
        Tensor downProjected;
        Tensor downCache = cache.get(downProjection);
        
        int seqLength = input.shapeAt(1);
        
        if (downCache == null) {
            Tensor upProjected = upProjection.forward(cache, norm2)[0].activateGrad(activation);
            downProjected = downProjection.forward(cache, upProjected)[0];
        } else {
            Range[] ranges = { Range.all(), Range.point(seqLength - 1), Range.all() };
            Tensor sliced = norm2.sliceGrad(ranges);
            
            Tensor upProj = upProjection.forward(cache, sliced)[0];
            Tensor activated = upProj.activateGrad(activation);
            Tensor downProj = downProjection.forward(cache, activated)[0];
            
            downProjected = downCache.concatGrad(downProj, 1);
        }
        
        cache.set(downProjection, downProjected);
        
        if (cache.isTraining()) {
            downProjected = dropout.forward(cache, downProjected)[0];
        }
        
        Tensor added2 = downProjected.addGrad(added);
        
        return new Tensor[] { added2 };
    }
    
    @Override
    public TransformerDecoder copy() {
        TransformerDecoder copy = new TransformerDecoder(numHeads, embeddingDim, dropoutRate);
        
        copy.normalizer1 = normalizer1.copy();
        copy.normalizer2 = normalizer2.copy();
        copy.upProjection = (DenseLayer) upProjection.copy();
        copy.downProjection = (DenseLayer) downProjection.copy();
        copy.attention = (MultiHeadAttention) attention.copy();
        copy.dropout = new DropoutLayer(dropoutRate);
        
        if (useGating) {
            copy.gateProjection = (DenseLayer) gateProjection.copy();
        }
        
        copy.useGating = useGating;
        copy.projDim = projDim;
        copy.attnQkvHasBias = attnQkvHasBias;
        copy.attnOutHasBias = attnOutHasBias;
        copy.normType = normType;
        
        return copy;
    }
}
