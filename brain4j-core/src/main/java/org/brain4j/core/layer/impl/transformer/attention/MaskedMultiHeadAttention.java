package org.brain4j.core.layer.impl.transformer.attention;

import org.brain4j.math.Tensors;
import org.brain4j.math.activation.impl.Softmax;
import org.brain4j.math.commons.Range;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.GpuTensor;

public class MaskedMultiHeadAttention extends MultiHeadAttention {
    
    public MaskedMultiHeadAttention(int headCount, int modelDimension) {
        super(headCount, modelDimension);
    }

    public MaskedMultiHeadAttention(Config config) {
        super(config);
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

        Tensor QKV = cachedProjection(cache, weights, slicingRanges, cachedQKV, input);

        if (config.qkvBias()) QKV = QKV.addGrad(bias);
        
        int D = config.embedDim();
        int H = config.heads();
        int d = headDim;
        
        Range all = Range.all();
        Tensor Q = QKV.sliceGrad(all, all, Range.interval(0, D));
        Tensor K = QKV.sliceGrad(all, all, Range.interval(D, 2 * D));
        Tensor V = QKV.sliceGrad(all, all, Range.interval(2 * D, 3 * D));
        
        Q = Q.reshapeGrad(batch, seqLength, H, d).transposeGrad(1, 2);
        K = K.reshapeGrad(batch, seqLength, H, d).transposeGrad(1, 2);
        V = V.reshapeGrad(batch, seqLength, H, d).transposeGrad(1, 2);
        
        double normalizer = Math.sqrt(headDim);
        
        Tensor mask = Tensors.triangularMask(seqLength, seqLength);
        
        if (input instanceof GpuTensor gpu) mask = mask.to(gpu.getDevice());
        
        Tensor K_T = K.transposeGrad();
        Tensor scores = Q.matmulGrad(K_T).divGrad(Tensors.scalar(normalizer));
        Tensor attentionMap = scores.addGrad(mask);
        Tensor probabilities = attentionMap.activateGrad(new Softmax());
        Tensor context = probabilities.matmulGrad(V);
        context = context.transposeGrad(1, 2);
        
        Tensor output = context.reshapeGrad(batch, seqLength, config.embedDim());
        Tensor result = cachedProjection(cache, outProj, slicingRanges, cachedOutput, output);

        if (config.outBias()) result = result.addGrad(outBias);
        
        return new Tensor[]{result};
    }

    private Tensor cachedProjection(StatesCache cache, Tensor weights, Range[] range, Tensor cachedOutput, Tensor output) {
        Tensor result;

        if (cachedOutput != null && !cache.isTraining()) {
            Tensor newOutput = output.slice(range);
            Tensor proj = newOutput.matmul(weights);

            result = cachedOutput.concat(proj, 1);
        } else {
            result = output.matmulGrad(weights);
        }

        cache.set(weights, result);
        return result;
    }
}
