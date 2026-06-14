package org.brain4j.core.transformer.attention;

import org.brain4j.core.layer.old.impl.transformer.MultiHeadAttention;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.impl.Softmax;
import org.brain4j.math.clipper.GradientClipper;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.GpuTensor;
import org.brain4j.math.commons.Range;

/**
 * A variant of multi-head attention that uses causal (triangular) masking.
 *
 * <p>This attention mechanism ensures that each position can only attend to
 * previous positions in the sequence, which is essential for autoregressive
 * models like GPT. It achieves this by adding a triangular mask with negative
 * infinity values to the attention scores before softmax.
 *
 * <p>The masking pattern looks like this for a sequence of length 4:
 * <pre>
 *  0  -∞  -∞  -∞
 *  0   0  -∞  -∞
 *  0   0   0  -∞
 *  0   0   0   0
 * </pre>
 *
 * <p>This ensures that when generating text, each token can only see
 * past tokens, not future ones.
 */
public class MaskedMultiHeadAttention extends MultiHeadAttention {

    public MaskedMultiHeadAttention(GradientClipper clipper, int headCount, int modelDimension) {
        super(clipper, headCount, modelDimension);
    }

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor input = inputs[0];
        int batch = input.shapeAt(0);
        int seqLength = input.shapeAt(1);

        

        Range[] slicingRanges = {
            Range.all(), Range.point(seqLength - 1), Range.all()
        }; // [batch, 1, dim]
        Tensor cachedOutput = cache.get(outProj);
        Tensor cachedQKV = cache.get(weights);
        Tensor QKV; // [batch, seq_len, 3 * H * head_dim]

        if (cachedQKV != null && !cache.isTraining()) {
            Tensor newTokens = input.slice(slicingRanges);
            Tensor proj = newTokens.matmul(weights);

            QKV = cachedQKV.concat(proj, 1);
        } else QKV = input.matmulGrad(weights);

        cache.set(weights, QKV);

        if (attnQkvHasBias) QKV = QKV.addGrad(bias);

        int D = embeddingDim;
        int H = headCount;
        int d = headDimension;

        Range all = Range.all();
        Tensor Q = QKV.sliceGrad(all, all, Range.interval(0, D));
        Tensor K = QKV.sliceGrad(all, all, Range.interval(D, 2 * D));
        Tensor V = QKV.sliceGrad(all, all, Range.interval(2 * D, 3 * D));

        // [batch, heads, seq_len, head_dim]
        Q = Q.reshapeGrad(batch, seqLength, H, d).transposeGrad(1, 2);
        K = K.reshapeGrad(batch, seqLength, H, d).transposeGrad(1, 2);
        V = V.reshapeGrad(batch, seqLength, H, d).transposeGrad(1, 2);

        double normalizer = Math.sqrt(headDimension);

        Tensor mask = Tensors.triangularMask(seqLength, seqLength);

        if (input instanceof GpuTensor gpu) mask = mask.to(gpu.getDevice());

        // [batch, heads, head_dim, seq_len]
        Tensor K_T = K.transposeGrad();
        // [batch, heads, seq_len, seq_len]
        Tensor scores = Q.matmulGrad(K_T).div(normalizer);
        Tensor attentionMap = scores.addGrad(mask);
        Tensor probabilities = attentionMap.activateGrad(new Softmax());
        // [batch, heads, seq_len, head_dim]
        Tensor context = probabilities.matmulGrad(V);
        // [batch, seq_len, heads, head_dim]
        context = context.transposeGrad(1, 2);

        // [batch, seq_len, embedding_dim]
        Tensor output = context.reshapeGrad(batch, seqLength, embeddingDim);
        Tensor result;

        if (cachedOutput != null && !cache.isTraining()) {
            Tensor newOutput = output.slice(slicingRanges);
            Tensor proj = newOutput.matmul(outProj);

            result = cachedOutput.concat(proj, 1);
        } else result = output.matmulGrad(outProj);

        cache.set(outProj, result);

        if (attnOutHasBias) result = result.addGrad(outBias);

        return new Tensor[]{result};
    }
}
