package org.brain4j.core.layer.newimpl.transformer;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.impl.Linear;
import org.brain4j.math.activation.impl.Softmax;
import org.brain4j.math.clipper.GradientClipper;
import org.brain4j.math.clipper.impl.HardClipper;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.commons.Range;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.gpu.ops.FlashAttention;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.GpuTensor;

import java.util.List;
import java.util.random.RandomGenerator;

public class MultiHeadAttention extends Layer {
    
    protected int headCount;
    protected int embeddingDim;
    protected int headDimension;
    protected boolean attnQkvHasBias;
    protected boolean attnOutHasBias;
    protected boolean flashAttention;
    
    public MultiHeadAttention(int headCount, int embeddingDim) {
        this(new HardClipper(5), headCount, embeddingDim);
    }
    
    public MultiHeadAttention(GradientClipper clipper, int headCount, int embeddingDim) {
        super(new Linear());
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
    public void build(List<Shape> inputShapes) {
        Shape inputShape = inputShapes.getFirst();
        
        if (inputShape.rank() != 2) {
            throw Commons.illegalArgument("Attention requires tensors with rank 2 but %s were given!", inputShape.rank());
        }
        
        if (inputShape.last() != embeddingDim) {
            throw Commons.illegalArgument("Expected embedding dim %s but got %s", embeddingDim, inputShape.last());
        }
        
        registerParam("weights", Tensors.zeros(embeddingDim, 3 * embeddingDim));
        registerParam("out_proj", Tensors.zeros(embeddingDim, embeddingDim));
        
        if (attnQkvHasBias) registerParam("bias", Tensors.zeros(3 * embeddingDim));
        if (attnOutHasBias) registerParam("out_bias", Tensors.zeros(embeddingDim));
    }

    @Override
    public void initWeights(List<Shape> inputShapes, RandomGenerator rng) {
        generateWeights("weights", rng, embeddingDim, 3 * embeddingDim);
        generateWeights("out_proj", rng, embeddingDim, embeddingDim);
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
        
        if (flashAttention && input instanceof GpuTensor) {
            int H = headCount;
            int d = headDimension;
            
            boolean training = input.usesGrad();
            
            Tensor QKV = training ? input.matmulGrad(weights) : input.matmul(weights);
            if (attnQkvHasBias) QKV = training ? QKV.addGrad(bias) : QKV.add(bias);
            
            Range all = Range.all();
            Tensor Q, K, V;
            
            if (training) {
                Tensor reshaped = QKV.reshapeGrad(batch, seqLength, H, 3, d)
                    .transposeGrad(1, 2);
                Q = reshaped.sliceGrad(all, all, all, Range.point(0), all).squeezeGrad(3);
                K = reshaped.sliceGrad(all, all, all, Range.point(1), all).squeezeGrad(3);
                V = reshaped.sliceGrad(all, all, all, Range.point(2), all).squeezeGrad(3);
            } else {
                Tensor reshaped = QKV.reshape(batch, seqLength, H, 3, d)
                    .transpose(1, 2);
                Q = reshaped.slice(all, all, all, Range.point(0), all).squeezeGrad(3);
                K = reshaped.slice(all, all, all, Range.point(1), all).squeezeGrad(3);
                V = reshaped.slice(all, all, all, Range.point(2), all).squeezeGrad(3);
            }
            
            float scale = (float) (1.0 / Math.sqrt(d));
            
            Tensor context;
            if (training) {
                Tensor[] flashResult = FlashAttention.forwardWithLse(Q, K, V, scale, false);
                if (flashResult != null) {
                    context = flashResult[0];
                    cache.set(this, flashResult[1]);
                } else {
                    context = null;
                }
            } else {
                context = FlashAttention.forward(Q, K, V, scale, false);
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
        
        Tensor QKV = input.matmulGrad(weights);
        
        if (attnQkvHasBias) QKV = QKV.addGrad(bias);
        
        Tensor reshaped = QKV.reshapeGrad(batch, seqLength, headCount, 3, headDimension);
        reshaped = reshaped.transposeGrad(1, 2);
        
        Tensor[] QKVs = new Tensor[3];
        Range all = Range.all();
        
        for (int i = 0; i < QKVs.length; i++) {
            QKVs[i] = reshaped.sliceGrad(all, all, all, Range.point(i), all);
            QKVs[i] = QKVs[i].squeezeGrad(3);
        }
        
        Tensor Q = QKVs[0], K = QKVs[1], V = QKVs[2];
        
        double normalizer = Math.sqrt(headDimension);
        
        Tensor K_T = K.transposeGrad();
        Tensor scores = Q.matmulGrad(K_T).div(normalizer);
        Tensor attentionWeights = scores.activateGrad(new Softmax());
        Tensor context = attentionWeights.matmulGrad(V);
        context = context.transposeGrad(1, 2);
        Tensor output = context.reshapeGrad(batch, seqLength, embeddingDim);
        
        Tensor result = output.matmulGrad(outProj);
        
        if (attnOutHasBias) result = result.addGrad(outBias);
        
        return new Tensor[]{result};
    }

    @Override
    public Layer copy() {
        MultiHeadAttention copy = new MultiHeadAttention(clipper, headCount, embeddingDim);
        copy.attnQkvHasBias = attnQkvHasBias;
        copy.attnOutHasBias = attnOutHasBias;
        copy.flashAttention = flashAttention;
        copyParameters(copy);
        return copy;
    }
    
    public int headCount() {
        return headCount;
    }
    
    public int embeddingDim() {
        return embeddingDim;
    }
    
    public int headDimension() {
        return headDimension;
    }
    
    public boolean flashAttention() {
        return flashAttention;
    }
    
    public boolean attnQkvHasBias() {
        return attnQkvHasBias;
    }
    
    public boolean attnOutHasBias() {
        return attnOutHasBias;
    }
    
    public MultiHeadAttention flashAttention(boolean flashAttention) {
        this.flashAttention = flashAttention;
        return this;
    }
    
    public MultiHeadAttention attnQkvHasBias(boolean attnQkvHasBias) {
        this.attnQkvHasBias = attnQkvHasBias;
        return this;
    }
    
    public MultiHeadAttention attnOutHasBias(boolean attnOutHasBias) {
        this.attnOutHasBias = attnOutHasBias;
        return this;
    }
}
