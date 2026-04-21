package org.brain4j.core.layer.impl.transformer;

import org.brain4j.math.Tensors;
import org.brain4j.math.activation.impl.Softmax;
import org.brain4j.math.clipper.GradientClipper;
import org.brain4j.math.commons.Range;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.GpuTensor;

public class MaskedMultiHeadAttention extends MultiHeadAttention {
    
    public MaskedMultiHeadAttention(GradientClipper clipper, int headCount, int modelDimension) {
        super(clipper, headCount, modelDimension);
    }

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor input = inputs[0];
        int batch = input.shapeAt(0);
        int seqLength = input.shapeAt(1);
        
        Tensor weights = getParam("weights");
        Tensor outProj = getParam("out_proj");
        Tensor bias = getParam("bias");
        Tensor outBias = getParam("out_bias");
        
        Range[] slicingRanges = {
            Range.all(), Range.point(seqLength - 1), Range.all()
        };
        Tensor cachedOutput = cache.get(outProj);
        Tensor cachedQKV = cache.get(weights);
        Tensor QKV;
        
        if (cachedQKV != null && !cache.isTraining()) {
            Tensor newTokens = input.slice(slicingRanges);
            Tensor proj = newTokens.matmul(weights);
            QKV = cachedQKV.concat(proj, 1);
        } else {
            QKV = input.matmulGrad(weights);
        }
        
        cache.set(weights, QKV);
        
        if (attnQkvHasBias) QKV = QKV.addGrad(bias);
        
        int D = embeddingDim;
        int H = headCount;
        int d = headDimension;
        
        Range all = Range.all();
        Tensor Q = QKV.sliceGrad(all, all, Range.interval(0, D));
        Tensor K = QKV.sliceGrad(all, all, Range.interval(D, 2 * D));
        Tensor V = QKV.sliceGrad(all, all, Range.interval(2 * D, 3 * D));
        
        Q = Q.reshapeGrad(batch, seqLength, H, d).transposeGrad(1, 2);
        K = K.reshapeGrad(batch, seqLength, H, d).transposeGrad(1, 2);
        V = V.reshapeGrad(batch, seqLength, H, d).transposeGrad(1, 2);
        
        double normalizer = Math.sqrt(headDimension);
        
        Tensor mask = Tensors.triangularMask(seqLength, seqLength);
        
        if (input instanceof GpuTensor gpu) mask = mask.to(gpu.device());
        
        Tensor K_T = K.transposeGrad();
        Tensor scores = Q.matmulGrad(K_T).div(normalizer);
        Tensor attentionMap = scores.addGrad(mask);
        Tensor probabilities = attentionMap.activateGrad(new Softmax());
        Tensor context = probabilities.matmulGrad(V);
        context = context.transposeGrad(1, 2);
        
        Tensor output = context.reshapeGrad(batch, seqLength, embeddingDim);
        Tensor result;
        
        if (cachedOutput != null && !cache.isTraining()) {
            Tensor newOutput = output.slice(slicingRanges);
            Tensor proj = newOutput.matmul(outProj);
            
            result = cachedOutput.concat(proj, 1);
        } else {
            result = output.matmulGrad(outProj);
        }
        
        cache.set(outProj, result);
        
        if (attnOutHasBias) result = result.addGrad(outBias);
        
        return new Tensor[]{result};
    }
}
