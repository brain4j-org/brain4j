package org.brain4j.core.layer.impl.transformer.attention;

import org.brain4j.math.Tensors;
import org.brain4j.math.activation.impl.Softmax;
import org.brain4j.math.commons.Range;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.GpuTensor;

/**
 * Applies causal multi-head attention with key-value caching.
 * Use it in decoders for autoregressive generation.
 */
public class MaskedMultiHeadAttention extends MultiHeadAttention {

    /**
     * Create attention without biases.
     */
    public MaskedMultiHeadAttention(int headCount, int modelDimension) {
        super(headCount, modelDimension);
    }

    /**
     * Create attention with the given config.
     */
    public MaskedMultiHeadAttention(Config config) {
        super(config);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor input = inputs[0];
        int batch = input.shapeAt(0);
        int seqLength = input.shapeAt(1);

        Tensor weights = param("weights");
        Tensor outProj = param("out_proj");
        Tensor bias = param("bias");
        Tensor outBias = param("out_bias");

        Range[] slicingRanges = {
            Range.all(), Range.point(seqLength - 1), Range.all()
        };
        Tensor cachedOutput = cache.get(outProj);
        Tensor cachedQkv = cache.get(weights);

        Tensor qkv = cachedProjection(cache, weights, slicingRanges, cachedQkv, input);

        if (config.qkvBias()) {
            qkv = qkv.add(bias);
        }

        int d = headDim;
        int heads = config.heads();
        int dim = config.embedDim();

        Range all = Range.all();
        Tensor q = qkv.slice(all, all, Range.interval(0, dim));
        Tensor k = qkv.slice(all, all, Range.interval(dim, 2 * dim));
        Tensor v = qkv.slice(all, all, Range.interval(2 * dim, 3 * dim));

        q = q.reshape(batch, seqLength, heads, d).transpose(1, 2);
        k = k.reshape(batch, seqLength, heads, d).transpose(1, 2);
        v = v.reshape(batch, seqLength, heads, d).transpose(1, 2);

        double normalizer = Math.sqrt(headDim);

        Tensor mask = Tensors.triangularMask(seqLength, seqLength);

        if (input instanceof GpuTensor gpu) {
            mask = mask.to(gpu.device());
        }

        Tensor kT = k.transpose();
        Tensor scores = q.matmul(kT).div(Tensors.scalar(normalizer));
        Tensor attentionMap = scores.add(mask);
        Tensor probabilities = attentionMap.activate(new Softmax());
        Tensor context = probabilities.matmul(v);
        context = context.transpose(1, 2);

        Tensor output = context.reshape(batch, seqLength, config.embedDim());
        Tensor result = cachedProjection(cache, outProj, slicingRanges, cachedOutput, output);

        if (config.outBias()) {
            result = result.add(outBias);
        }

        return new Tensor[]{result};
    }

    private Tensor cachedProjection(StatesCache cache, Tensor weights, Range[] range, Tensor cachedOutput, Tensor output) {
        Tensor result;

        if (cachedOutput != null && !cache.isTraining()) {
            Tensor newOutput = output.slice(range);
            Tensor proj = newOutput.matmul(weights);

            result = cachedOutput.concat(proj, 1);
        } else {
            result = output.matmul(weights);
        }

        cache.set(weights, result);

        return result;
    }
}
