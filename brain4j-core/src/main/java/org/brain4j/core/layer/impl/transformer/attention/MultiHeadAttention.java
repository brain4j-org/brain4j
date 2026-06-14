package org.brain4j.core.layer.impl.transformer.attention;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.impl.Softmax;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.commons.Range;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

public class MultiHeadAttention extends Layer {
    
    protected int headDim;
    protected Config config;

    public record Config(int embedDim, int heads, boolean qkvBias, boolean outBias) {}

    public MultiHeadAttention(int heads, int embedDim) {
        this(new Config(embedDim, heads, false, false));
    }

    public MultiHeadAttention(Config config) {
        if (config.embedDim % config.heads != 0) {
            throw Commons.illegalArgument(
                    "Embedding dimension must be divisible by head count! (%s %% %s = %s)",
                    config.embedDim, config.heads, config.embedDim % config.heads
            );
        }
        
        this.headDim = config.embedDim / config.heads;
        this.config = config;
    }

    @Override
    public void build(List<Shape> inputShapes) {
        Shape inputShape = inputShapes.getFirst();
        
        if (inputShape.rank() != 2) {
            throw Commons.illegalArgument("Attention requires tensors with rank 2 but %s were given!", inputShape.rank());
        }
        
        if (inputShape.last() != config.embedDim) {
            throw Commons.illegalArgument("Expected embedding dim %s but got %s", config.embedDim, inputShape.last());
        }
        
        registerParam("weights", Tensors.zeros(config.embedDim, 3 * config.embedDim));
        registerParam("out_proj", Tensors.zeros(config.embedDim, config.embedDim));
        
        if (config.qkvBias) registerParam("bias", Tensors.zeros(3 * config.embedDim));
        if (config.outBias) registerParam("out_bias", Tensors.zeros(config.embedDim));
    }

    @Override
    public void initWeights(List<Shape> inputShapes, RandomGenerator rng) {
        generateWeights("weights", rng, config.embedDim, 3 * config.embedDim);
        generateWeights("out_proj", rng, config.embedDim, config.embedDim);
    }

    @Override
    public List<Shape> inferOutputShapes(List<Shape> inputShapes) {
        if (inputShapes.size() != 1) {
            throw Commons.illegalArgument("Layer requires 1 input but %s were given!", inputShapes.size());
        }
        
        Shape inputShape = inputShapes.getFirst();
        
        if (inputShape.rank() != 2) {
            throw Commons.illegalArgument("Attention requires tensors with rank 2 but %s were given!", inputShape.rank());
        }
        
        return List.of(inputShape);
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
        Tensor QKV = input.matmulGrad(weights);
        
        if (config.qkvBias) QKV = QKV.addGrad(bias);
        
        Tensor reshaped = QKV.reshapeGrad(batch, seqLength, config.heads, 3, headDim);
        reshaped = reshaped.transposeGrad(1, 2);
        
        Tensor[] QKVs = new Tensor[3];
        Range all = Range.all();
        
        for (int i = 0; i < QKVs.length; i++) {
            QKVs[i] = reshaped.sliceGrad(all, all, all, Range.point(i), all);
            QKVs[i] = QKVs[i].squeezeGrad(3);
        }
        
        Tensor Q = QKVs[0], K = QKVs[1], V = QKVs[2];
        
        double normalizer = Math.sqrt(headDim);
        
        Tensor K_T = K.transposeGrad();
        Tensor scores = Q.matmulGrad(K_T).div(normalizer);
        Tensor attentionWeights = scores.activateGrad(new Softmax());
        Tensor context = attentionWeights.matmulGrad(V);
        context = context.transposeGrad(1, 2);
        Tensor output = context.reshapeGrad(batch, seqLength, config.embedDim);
        
        Tensor result = output.matmulGrad(outProj);
        
        if (config.outBias) result = result.addGrad(outBias);
        
        return new Tensor[]{result};
    }

    @Override
    public Layer copy() {
        MultiHeadAttention copy = new MultiHeadAttention(config);
        copyParameters(copy);
        return copy;
    }

    public int headDim() {
        return headDim;
    }

    public Config config() {
        return config;
    }
}
