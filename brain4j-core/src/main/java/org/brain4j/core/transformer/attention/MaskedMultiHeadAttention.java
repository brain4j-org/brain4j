package org.brain4j.core.transformer.attention;

import org.brain4j.core.layer.impl.transformer.MultiHeadAttention;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.impl.SoftmaxActivation;
import org.brain4j.math.clipper.GradientClipper;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.gpu.ops.FlashAttention;
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

        if (flashAttention && input instanceof GpuTensor) {
            // skip fast path if we are using incremental cache
            Tensor cachedQKV = cache.get(weights);
            if (cachedQKV == null) {
                int H = headCount;
                int d = headDimension;
                boolean training = input.usesGrad();

                Tensor QKV = training ? input.matmulGrad(weights) : input.matmul(weights);
                if (attnQkvHasBias) QKV = training ? QKV.addGrad(bias) : QKV.add(bias);

                Range all = Range.all();
                Tensor Q, K, V;

                if (training) {
                    Tensor reshaped = QKV.reshapeGrad(batch, seqLength, H, d * 3);
                    Q = reshaped.sliceGrad(all, all, Range.interval(0, d * H))
                            .reshapeGrad(batch, seqLength, H, d)
                            .transposeGrad(1, 2); // [B,H,L,d]
                    K = reshaped.sliceGrad(all, all, Range.interval(d * H, 2 * d * H))
                            .reshapeGrad(batch, seqLength, H, d)
                            .transposeGrad(1, 2); // [B,H,L,d]
                    V = reshaped.sliceGrad(all, all, Range.interval(2 * d * H, 3 * d * H))
                            .reshapeGrad(batch, seqLength, H, d)
                            .transposeGrad(1, 2); // [B,H,L,d]
                } else {
                    Q = QKV.slice(all, all, Range.interval(0, embeddingDim))
                            .reshape(batch, seqLength, H, d)
                            .transpose(1, 2); // [B,H,L,d]
                    K = QKV.slice(all, all, Range.interval(embeddingDim, 2 * embeddingDim))
                            .reshape(batch, seqLength, H, d)
                            .transpose(1, 2); // [B,H,L,d]
                    V = QKV.slice(all, all, Range.interval(2 * embeddingDim, 3 * embeddingDim))
                            .reshape(batch, seqLength, H, d)
                            .transpose(1, 2); // [B,H,L,d]
                }

                float scale = (float) (1.0 / Math.sqrt(d));

                Tensor context;
                if (training) {
                    // use forward with LSE for training (required for backward pass)
                    Tensor[] flashResult = FlashAttention.forwardWithLse(Q, K, V, scale, true);
                    if (flashResult != null) {
                        context = flashResult[0];
                        // store LSE in cache for potential use in backward
                        cache.set(this, flashResult[1]);
                    } else {
                        context = null;
                    }
                } else {
                    context = FlashAttention.forward(Q, K, V, scale, true);
                }

                if (context != null) {
                    Tensor output = training
                        ? context.transposeGrad(1, 2).reshapeGrad(batch, seqLength, embeddingDim)
                        : context.transpose(1, 2).reshape(batch, seqLength, embeddingDim);

                    Tensor result = training
                        ? output.matmulGrad(outProj)
                        : output.matmul(outProj);

                    if (attnOutHasBias) {
                        result = training ? result.addGrad(outBias) : result.add(outBias);
                    }
                    return new Tensor[]{result};
                }
            }
            // else fall through to standard path
        }

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

        if (input instanceof GpuTensor gpu) mask = mask.to(gpu.device());

        // [batch, heads, head_dim, seq_len]
        Tensor K_T = K.transposeGrad();
        // [batch, heads, seq_len, seq_len]
        Tensor scores = Q.matmulGrad(K_T).div(normalizer);
        Tensor attentionMap = scores.addGrad(mask);
        Tensor probabilities = attentionMap.activateGrad(new SoftmaxActivation());
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
