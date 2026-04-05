package org.brain4j.core.layer.impl.transformer;

import org.brain4j.core.layer.OldLayer;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.impl.Softmax;
import org.brain4j.math.clipper.GradientClipper;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.gpu.silicon.SiliconDevice;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.commons.Range;

import java.util.random.RandomGenerator;

/**
 * Implements the Multi-Head Attention mechanism as used in Transformer architectures.
 * <p>
 * This class manages the projection of input embeddings into queries (Q), keys (K), and values (V),
 * distributes them across multiple attention heads, computes attention scores using scaled dot-product attention,
 * and combines the outputs into a single representation. It also supports gradient clipping, weight initialization,
 * device placement, and optimization via {@link Optimizer} and {@link Updater}.
 * </p>
 * <h2>Shape conventions:</h2>
 * <ul>
 *   <li>Input: {@code [batch, seq_len, embedding_dim]}</li>
 *   <li>Q, K, V: {@code [batch, heads, seq_len, head_dim]}</li>
 *   <li>Attention scores: {@code [batch, heads, seq_len, seq_len]}</li>
 *   <li>Output: {@code [batch, seq_len, embedding_dim]}</li>
 * </ul>
 *
 * @author xEcho1337
 * @see Tensor
 * @see Optimizer
 * @see Updater
 */
public class MultiHeadAttention extends OldLayer {

    protected Tensor outProj;
    protected Tensor outBias;
    protected int headCount;
    protected int embeddingDim;
    protected int headDimension;
    protected boolean attnQkvHasBias;
    protected boolean attnOutHasBias;

    private MultiHeadAttention() {
    }

    /**
     * Configures the Multi-Head attention mechanism with the specified parameters.
     *
     * @param clipper      the gradient clipper to use during training
     * @param headCount    the amount of heads in the MHA; MUST be a multiple of the embedding dimension
     * @param embeddingDim the embedding dimension of the transformer
     */
    public MultiHeadAttention(GradientClipper clipper, int headCount, int embeddingDim) {
        this.clipper = clipper;
        this.headCount = headCount;
        this.embeddingDim = embeddingDim;
        this.attnQkvHasBias = true;

        if (embeddingDim % headCount != 0) {
            throw Commons.illegalArgument("Embedding dimension must be divisible by head count! (%s %% %s = %s)",
                embeddingDim, headCount, embeddingDim % headCount);
        }

        this.headDimension = embeddingDim / headCount;
    }

    @Override
    public void connect() {
        this.outProj = Tensors.matrix(embeddingDim, embeddingDim).withGrad();

        if (attnQkvHasBias) this.bias = Tensors.zeros(3 * embeddingDim).withGrad();
        if (attnOutHasBias) this.outBias = Tensors.zeros(embeddingDim).withGrad();

    }

    @Override
    public void initWeights(RandomGenerator generator, int input, int output) {
        Tensor keyProj = Tensors.zeros(embeddingDim, embeddingDim)
                .map(x -> weightInit.generate(generator, embeddingDim, embeddingDim));
        Tensor queryProj = Tensors.zeros(embeddingDim, embeddingDim)
                .map(x -> weightInit.generate(generator, embeddingDim, embeddingDim));
        Tensor valueProj = Tensors.zeros(embeddingDim, embeddingDim)
                .map(x -> weightInit.generate(generator, embeddingDim, embeddingDim));

        this.outProj.map(x -> weightInit.generate(generator, embeddingDim, embeddingDim));
        this.weights = Tensors.concat(keyProj, queryProj, valueProj).withGrad();
    }

    /**
     * Computes the forward pass of the multi-head attention mechanism for a given input tensor.
     * <p>This implementation follows the original architecture defined
     * by the <a href="https://arxiv.org/abs/1706.03762">original paper</a>.
     *
     * @param cache  cache used to store intermediate results
     * @param inputs the input tensors, first tensor must have shape {@code [batch, seq_len, embedding_dim]}
     * @return the output tensor of shape {@code [batch, seq_len, embedding_dim]}
     */
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor input = inputs[0];
        int batch = input.shapeAt(0);
        int seqLength = input.shapeAt(1);

        // [batch, seq_len, 3 * H * head_dim]
        Tensor QKV = input.matmulGrad(weights);

        if (attnQkvHasBias) QKV = QKV.addGrad(bias);

        Tensor reshaped = QKV.reshapeGrad(batch, seqLength, headCount, 3, headDimension);

        // [batch, heads, seq_len, 3, head_dim]
        reshaped = reshaped.transposeGrad(1, 2);

        Tensor[] QKVs = new Tensor[3];
        Range all = Range.all();

        for (int i = 0; i < QKVs.length; i++) {
            // [batch, heads, seq_len, 1, head_dim]
            QKVs[i] = reshaped.sliceGrad(all, all, all, Range.point(i), all);
            // [batch, heads, seq_len, head_dim]
            QKVs[i] = QKVs[i].squeezeGrad(3);
        }

        // [batch, heads, seq_len, head_dim]
        Tensor Q = QKVs[0], K = QKVs[1], V = QKVs[2];

        double normalizer = Math.sqrt(headDimension);

        // [batch, heads, head_dim, seq_len]
        Tensor K_T = K.transposeGrad();
        // [batch, heads, seq_len, seq_len]
        Tensor scores = Q.matmulGrad(K_T).div(normalizer);
        Tensor attentionWeights = scores.activateGrad(new Softmax());
        // [batch, heads, seq_len, head_dim]
        Tensor context = attentionWeights.matmulGrad(V);
        // [batch, seq_len, heads, head_dim]
        context = context.transposeGrad(1, 2);
        // [batch, seq_len, embedding_dim]
        Tensor output = context.reshapeGrad(batch, seqLength, embeddingDim);
        // [batch, seq_len, embedding_dim]

        Tensor result = output.matmulGrad(outProj);

        if (attnOutHasBias) result = result.addGrad(outBias);

        return new Tensor[]{result};
    }

    @Override
    public void backward(StatesCache cache, Updater updater, Optimizer optimizer) {
        super.backward(cache, updater, optimizer);

        Tensor optimized = optimizer.step(outProj);
        clipper.clip(optimized);
        updater.change(outProj, optimized);

        if (attnOutHasBias) {
            Tensor biasGrad = outBias.grad().sum(0, false);
            clipper.clip(biasGrad);
            updater.change(outBias, biasGrad);
        }
    }

    @Override
    public void toDevice(SiliconDevice device) {
        this.weights = toPersistentTensor(weights, device);
        this.outProj = toPersistentTensor(outProj, device);
        if (attnQkvHasBias) this.bias = toPersistentTensor(bias, device);
        if (attnOutHasBias) this.outBias = toPersistentTensor(outBias, device);
    }

    @Override
    public OldLayer freeze() {
        outProj.noGrad();
        if (outBias != null) outBias.noGrad();
        return super.freeze();
    }

    @Override
    public OldLayer unfreeze() {
        outProj.withGrad();
        if (outBias != null) outBias.withGrad();
        return super.unfreeze();
    }

    @Override
    public int size() {
        return embeddingDim;
    }

    @Override
    public int getTotalWeights() {
        return weights.elements() + outProj.elements();
    }

    @Override
    public int getTotalBias() {
        int total = 0;
        if (attnQkvHasBias) total += bias.elements();
        if (attnOutHasBias) total += outBias.elements();
        return total;
    }
    
    public void resetGrad() {
        super.resetGrad();
        outProj.zeroGrad();
        if (attnOutHasBias) outBias.zeroGrad();
    }

    public Tensor getOutProj() {
        return outProj;
    }

    public void setOutProj(Tensor outProj) {
        this.outProj = outProj;
    }

    public Tensor getOutBias() {
        return outBias;
    }

    public void setOutBias(Tensor outBias) {
        this.outBias = outBias;
    }

    public int getHeadCount() {
        return headCount;
    }

    public MultiHeadAttention setHeadCount(int headCount) {
        this.headCount = headCount;
        return this;
    }

    public int getEmbeddingDim() {
        return embeddingDim;
    }

    public MultiHeadAttention setEmbeddingDim(int embeddingDim) {
        this.embeddingDim = embeddingDim;
        return this;
    }

    public int getHeadDimension() {
        return headDimension;
    }

    public MultiHeadAttention setHeadDimension(int headDimension) {
        this.headDimension = headDimension;
        return this;
    }

    public boolean hasAttnQkvBias() {
        return attnQkvHasBias;
    }

    public MultiHeadAttention setAttnQkvBias(boolean attnQkvHasBias) {
        this.attnQkvHasBias = attnQkvHasBias;
        return this;
    }

    public boolean hasAttnOutBias() {
        return attnOutHasBias;
    }

    public MultiHeadAttention setAttnOutBias(boolean attnOutHasBias) {
        this.attnOutHasBias = attnOutHasBias;
        return this;
    }
}
