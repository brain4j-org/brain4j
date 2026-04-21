package org.brain4j.core.layer.impl.transformer;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.impl.DenseLayer;
import org.brain4j.core.layer.impl.DropoutLayer;
import org.brain4j.core.layer.impl.NormLayer;
import org.brain4j.core.layer.impl.RMSNormLayer;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.weightsinit.impl.UniformXavierInit;

import java.util.List;
import java.util.random.RandomGenerator;

public class TransformerEncoder extends Layer {
    
    protected DenseLayer upProjection;
    protected DenseLayer gateProjection;
    protected DenseLayer downProjection;
    protected Layer normalizer1;
    protected Layer normalizer2;
    protected DropoutLayer dropout;
    protected MultiHeadAttention attention;
    protected NormType normType;
    
    protected int numHeads;
    protected int embeddingDim;
    protected int projDim;
    protected double dropoutRate;
    protected boolean useGating;
    protected boolean attnQkvHasBias;
    protected boolean attnOutHasBias;
    
    public TransformerEncoder(int numHeads, int embeddingDim, double dropout) {
        this(numHeads, embeddingDim, dropout, Activations.GELU);
    }
    
    public TransformerEncoder(int numHeads, int embeddingDim, double dropout, Activations activation) {
        this(numHeads, embeddingDim, 4 * embeddingDim, dropout, false, false, false, activation.function(), NormType.LAYER_NORM);
    }
    
    public TransformerEncoder(
        int numHeads,
        int embeddingDim,
        int projDim,
        double dropout,
        boolean useGating,
        boolean attnQkvHasBias,
        boolean attnOutHasBias,
        Activation activation,
        NormType normType
    ) {
        super(activation);
        this.numHeads = numHeads;
        this.embeddingDim = embeddingDim;
        this.projDim = projDim;
        this.dropoutRate = dropout;
        this.normType = normType;
        this.useGating = useGating;
        this.attnQkvHasBias = attnQkvHasBias;
        this.attnOutHasBias = attnOutHasBias;
        
        this.dropout = new DropoutLayer(dropout);
        this.weightInit = new UniformXavierInit();
        this.normalizer1 = createNormLayer();
        this.normalizer2 = createNormLayer();
        this.upProjection = new DenseLayer(projDim);
        this.downProjection = new DenseLayer(embeddingDim);
        this.attention = createAttention(numHeads, embeddingDim);
        
        if (useGating) this.gateProjection = new DenseLayer(projDim);
        
        attention.attnQkvHasBias(attnQkvHasBias);
        attention.attnOutHasBias(attnOutHasBias);
    }
    
    protected Layer createNormLayer() {
        return switch (normType) {
            case LAYER_NORM -> new NormLayer();
            case RMS_NORM -> new RMSNormLayer();
        };
    }
    
    protected MultiHeadAttention createAttention(int heads, int embeddingDim) {
        return new MultiHeadAttention(clipper, heads, embeddingDim);
    }

    @Override
    public void build(List<Shape> inputShapes) {
        Shape inputShape = inputShapes.getFirst();
        Shape projShape = Shape.of(inputShape.dim(0), projDim);
        
        normalizer1.build(List.of(inputShape));
        normalizer2.build(List.of(inputShape));
        upProjection.build(List.of(inputShape));
        downProjection.build(List.of(projShape));
        attention.build(List.of(inputShape));
        
        if (useGating) gateProjection.build(List.of(inputShape));
    }

    @Override
    public void initWeights(List<Shape> inputShapes, RandomGenerator rng) {
        Shape inputShape = inputShapes.getFirst();
        Shape projShape = Shape.of(inputShape.dim(0), projDim);
        
        normalizer1.initWeights(List.of(inputShape), rng);
        normalizer2.initWeights(List.of(inputShape), rng);
        upProjection.initWeights(List.of(inputShape), rng);
        downProjection.initWeights(List.of(projShape), rng);
        attention.initWeights(List.of(inputShape), rng);
        
        normalizer1.initAutoGrad();
        normalizer2.initAutoGrad();
        upProjection.initAutoGrad();
        downProjection.initAutoGrad();
        attention.initAutoGrad();
        
        if (useGating) {
            gateProjection.initWeights(List.of(inputShape), rng);
            gateProjection.initAutoGrad();
        }
    }

    @Override
    public List<Shape> inferOutputShapes(List<Shape> inputShapes) {
        if (inputShapes.size() != 1) {
            throw Commons.illegalArgument("Layer requires 1 input but %s were given!", inputShapes.size());
        }
        
        Shape inputShape = inputShapes.getFirst();
        
        if (inputShape.rank() != 2) {
            throw Commons.illegalArgument("Transformer requires tensors with rank 2 but %s were given!", inputShape.rank());
        }
        
        if (inputShape.last() != embeddingDim) {
            throw Commons.illegalArgument("Expected embedding dim %s but got %s", embeddingDim, inputShape.last());
        }
        
        return List.of(inputShape);
    }

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor input = inputs[0];
        
        if (input.rank() != 3) {
            throw Commons.illegalArgument("Input must have shape [batch, seq_length, dimension]! Got: %s",
                java.util.Arrays.toString(input.shape()));
        }
        
        Tensor attended = attention.forward(cache, input)[0];
        
        if (cache.isTraining()) {
            attended = dropout.forward(cache, attended)[0];
        }
        
        Tensor added = attended.addGrad(input);
        Tensor normalized = normalizer1.forward(cache, added)[0];
        
        Tensor downProjected;
        
        if (gateProjection != null) {
            Tensor gate = gateProjection.forward(cache, normalized)[0].activateGrad(activation);
            Tensor up = upProjection.forward(cache, normalized)[0];
            Tensor prod = gate.mul(up);
            downProjected = downProjection.forward(cache, prod)[0];
        } else {
            Tensor upProjected = upProjection.forward(cache, normalized)[0].activateGrad(activation);
            downProjected = downProjection.forward(cache, upProjected)[0];
        }
        
        if (cache.isTraining()) {
            downProjected = dropout.forward(cache, downProjected)[0];
        }
        
        Tensor added2 = downProjected.addGrad(normalized);
        Tensor normalized2 = normalizer2.forward(cache, added2)[0];
        
        return new Tensor[] { normalized2 };
    }

    @Override
    public void backward(Updater updater, Optimizer optimizer) {
        normalizer2.backward(updater, optimizer);
        downProjection.backward(updater, optimizer);
        upProjection.backward(updater, optimizer);
        normalizer1.backward(updater, optimizer);
        attention.backward(updater, optimizer);
        
        if (useGating) {
            gateProjection.backward(updater, optimizer);
        }
    }
    
    @Override
    public void resetGrad() {
        normalizer1.resetGrad();
        normalizer2.resetGrad();
        upProjection.resetGrad();
        downProjection.resetGrad();
        attention.resetGrad();
        
        if (useGating) {
            gateProjection.resetGrad();
        }
    }
    
    @Override
    public Layer freeze() {
        normalizer1.freeze();
        normalizer2.freeze();
        upProjection.freeze();
        downProjection.freeze();
        attention.freeze();
        
        if (useGating) gateProjection.freeze();
        
        return super.freeze();
    }
    
    @Override
    public Layer unfreeze() {
        normalizer1.unfreeze();
        normalizer2.unfreeze();
        upProjection.unfreeze();
        downProjection.unfreeze();
        attention.unfreeze();
        
        if (useGating) gateProjection.unfreeze();
        
        return super.unfreeze();
    }

    @Override
    public Layer copy() {
        TransformerEncoder copy = new TransformerEncoder(
            numHeads, embeddingDim, projDim, dropoutRate, useGating,
            attnQkvHasBias, attnOutHasBias, activation, normType
        );
        
        copy.normalizer1 = normalizer1.copy();
        copy.normalizer2 = normalizer2.copy();
        copy.upProjection = (DenseLayer) upProjection.copy();
        copy.downProjection = (DenseLayer) downProjection.copy();
        copy.attention = (MultiHeadAttention) attention.copy();
        copy.dropout = new DropoutLayer(dropoutRate);
        
        if (useGating) {
            copy.gateProjection = (DenseLayer) gateProjection.copy();
        }
        
        return copy;
    }
    
    public int embeddingDim() {
        return embeddingDim;
    }
    
    public int numHeads() {
        return numHeads;
    }
    
    public double dropoutRate() {
        return dropoutRate;
    }
}
