package org.brain4j.core.layer.impl.transformer;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.impl.DenseLayer;
import org.brain4j.core.layer.impl.DropoutLayer;
import org.brain4j.core.layer.impl.NormLayer;
import org.brain4j.core.layer.impl.transformer.attention.MaskedMultiHeadAttention;
import org.brain4j.core.layer.impl.transformer.attention.MultiHeadAttention;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.impl.GELU;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.weightsinit.impl.UniformXavierInit;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;
import java.util.random.RandomGenerator;

public abstract class Transformer<T extends Transformer<T>> extends Layer {

    protected DenseLayer upProj;
    protected DenseLayer gateProj;
    protected DenseLayer downProj;
    protected DropoutLayer dropout;
    protected Layer norm1;
    protected Layer norm2;
    protected Layer attention;
    protected Config config;

    public record Config(
        int embedDim,
        int projDim,
        int heads,
        double dropout,
        boolean gating,
        Activation activation,
        Supplier<Layer> normSupplier
    ) {}

    public Transformer(int embedDim, int heads, double dropout) {
        this(embedDim, heads, dropout, new GELU());
    }

    public Transformer(int embedDim, int heads, double dropout, Activation activation) {
        this(new Config(embedDim, 4 * embedDim, heads, dropout, false, activation, NormLayer::new));
    }

    public Transformer(Config config) {
        this.config = config;

        this.dropout = new DropoutLayer(config.dropout);
        this.weightInit = new UniformXavierInit();
        this.norm1 = config.normSupplier.get();
        this.norm2 = config.normSupplier.get();
        this.upProj = new DenseLayer(config.projDim);
        this.downProj = new DenseLayer(config.embedDim);
        this.attention = getAttention();

        if (config.gating)
            this.gateProj = new DenseLayer(config.projDim);
    }

    protected abstract T getCopy();

    protected abstract Layer getAttention();

    public Config config() {
        return config;
    }

    @Override
    public void build(List<Shape> inputShapes) {
        Shape inputShape = inputShapes.getFirst();
        Shape projShape = Shape.of(inputShape.dim(0), config.projDim);

        norm1.build(List.of(inputShape));
        norm2.build(List.of(inputShape));
        upProj.build(List.of(inputShape));
        downProj.build(List.of(projShape));
        attention.build(List.of(inputShape));

        if (gateProj != null)
            gateProj.build(List.of(inputShape));
    }

    @Override
    public void initWeights(List<Shape> inputShapes, RandomGenerator rng) {
        Shape inputShape = inputShapes.getFirst();
        Shape projShape = Shape.of(inputShape.dim(0), config.projDim);

        norm1.initWeights(List.of(inputShape), rng);
        norm2.initWeights(List.of(inputShape), rng);
        upProj.initWeights(List.of(inputShape), rng);
        downProj.initWeights(List.of(projShape), rng);
        attention.initWeights(List.of(inputShape), rng);

        norm1.initAutoGrad();
        norm2.initAutoGrad();
        upProj.initAutoGrad();
        downProj.initAutoGrad();
        attention.initAutoGrad();

        if (gateProj != null) {
            gateProj.initWeights(List.of(inputShape), rng);
            gateProj.initAutoGrad();
        }
    }

    @Override
    public List<Shape> inferOutputShapes(List<Shape> inputShapes) {
        if (inputShapes.size() != 1) {
            throw Commons.illegalArgument("Layer requires (1) input but (%s) were given!", inputShapes.size());
        }

        Shape inputShape = inputShapes.getFirst();

        if (inputShape.rank() != 2) {
            throw Commons.illegalArgument("Layer requires input tensors with shape [N, D], got %s", inputShape);
        }

        if (inputShape.last() != config.embedDim) {
            throw Commons.illegalArgument("Expected embedding dim %s but got %s", config.embedDim, inputShape.last());
        }

        return List.of(inputShape);
    }

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor input = inputs[0];

        if (input.rank() != 3) {
            throw Commons.illegalArgument(
                    "Input must have shape [batch, seq_length, dimension]! Got: %s",
                    Arrays.toString(input.shape())
            );
        }

        Tensor x = input;

        x = norm1.forward(cache, x);
        x = attention.forward(cache, x);

        if (cache.isTraining()) {
            x = dropout.forward(cache, x);
        }

        x = input.addGrad(x);

        Tensor residual = x;

        x = norm2.forward(cache, x);

        Tensor y = upProj.forward(cache, x);

        if (gateProj != null) {
            x = gateProj.forward(cache, x);
            x = x.activateGrad(activation);
            x = x.mul(y);
        } else {
            x = y.activateGrad(activation);
        }

        x = downProj.forward(cache, x);

        if (cache.isTraining())
            x = dropout.forward(cache, x);

        x = residual.addGrad(x);

        return new Tensor[] { x };
    }

    @Override
    public void backward(Updater updater, Optimizer optimizer) {
        norm2.backward(updater, optimizer);
        downProj.backward(updater, optimizer);
        upProj.backward(updater, optimizer);
        norm1.backward(updater, optimizer);
        attention.backward(updater, optimizer);

        if (gateProj != null)
            gateProj.backward(updater, optimizer);
    }

    @Override
    public void resetGrad() {
        norm1.resetGrad();
        norm2.resetGrad();
        upProj.resetGrad();
        downProj.resetGrad();
        attention.resetGrad();

        if (gateProj != null)
            gateProj.resetGrad();
    }

    @Override
    public Layer freeze() {
        norm1.freeze();
        norm2.freeze();
        upProj.freeze();
        downProj.freeze();
        attention.freeze();

        if (gateProj != null)
            gateProj.freeze();

        return super.freeze();
    }

    @Override
    public Layer unfreeze() {
        norm1.unfreeze();
        norm2.unfreeze();
        upProj.unfreeze();
        downProj.unfreeze();
        attention.unfreeze();

        if (gateProj != null)
            gateProj.unfreeze();

        return super.unfreeze();
    }
    
    @Override
    public T copy() {
        T copy = getCopy();

        copy.norm1 = norm1.copy();
        copy.norm2 = norm2.copy();
        copy.upProj = upProj.copy();
        copy.downProj = downProj.copy();
        copy.attention = attention.copy();
        copy.dropout = dropout.copy();

        if (gateProj != null)
            copy.gateProj = gateProj.copy();

        return copy;
    }

    public static class Encoder extends Transformer<Encoder> {

        public Encoder(int embedDim, int heads, double dropout) {
            super(embedDim, heads, dropout);
        }

        public Encoder(int embedDim, int heads, double dropout, Activation activation) {
            super(embedDim, heads, dropout, activation);
        }

        public Encoder(Config config) {
            super(config);
        }

        @Override
        protected Encoder getCopy() {
            return new Encoder(config);
        }

        @Override
        protected Layer getAttention() {
            return new MultiHeadAttention(config.heads, config.embedDim);
        }
    }

    public static class Decoder extends Transformer<Decoder> {

        public Decoder(int embedDim, int heads, double dropout) {
            super(embedDim, heads, dropout);
        }

        public Decoder(int embedDim, int heads, double dropout, Activation activation) {
            super(embedDim, heads, dropout, activation);
        }

        public Decoder(Config config) {
            super(config);
        }

        @Override
        protected Decoder getCopy() {
            return new Decoder(config);
        }

        @Override
        protected Layer getAttention() {
            return new MaskedMultiHeadAttention(config.heads, config.embedDim);
        }
    }
}
