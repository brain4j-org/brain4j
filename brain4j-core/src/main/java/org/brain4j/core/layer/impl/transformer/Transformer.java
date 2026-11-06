package org.brain4j.core.layer.impl.transformer;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;
import java.util.random.RandomGenerator;

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

/**
 * Combines attention with a feed-forward block and residuals.
 * Use it as the base for encoder and decoder stacks.
 *
 * @param <T> the concrete transformer type
 */
public abstract class Transformer<T extends Transformer<T>> extends Layer {

    protected final Config config;
    protected DenseLayer upProj;
    protected DenseLayer gateProj;
    protected DenseLayer downProj;
    protected DropoutLayer dropout;
    protected Layer norm1;
    protected Layer norm2;
    protected Layer attention;

    /**
     * Holds embedding width, FFN width, heads, dropout, gating and norms.
     */
    public record Config(
        int embedDim,
        int projDim,
        int heads,
        double dropout,
        boolean gating,
        Activation activation,
        Supplier<Layer> normSupplier
    ) {}

    /**
     * Create a block with GELU activation.
     */
    public Transformer(int embedDim, int heads, double dropout) {
        this(embedDim, heads, dropout, new GELU());
    }

    /**
     * Create a block with the given activation.
     */
    public Transformer(int embedDim, int heads, double dropout, Activation activation) {
        this(new Config(embedDim, 4 * embedDim, heads, dropout, false, activation, NormLayer::new));
    }

    /**
     * Create a block with the given config.
     */
    public Transformer(Config config) {
        super(config.activation);
        this.config = config;

        this.dropout = new DropoutLayer(config.dropout);
        this.weightInit = new UniformXavierInit();
        this.norm1 = config.normSupplier.get();
        this.norm2 = config.normSupplier.get();
        this.upProj = new DenseLayer(config.projDim);
        this.downProj = new DenseLayer(config.embedDim);
        this.attention = attention();

        if (config.gating) {
            this.gateProj = new DenseLayer(config.projDim);
        }
    }

    /**
     * Create an empty copy with the same config.
     *
     * @return the new block
     */
    protected abstract T copyBlock();

    /**
     * Create the attention sublayer.
     *
     * @return the attention layer
     */
    protected abstract Layer attention();

    /**
     * Return the block config.
     *
     * @return the config
     */
    public Config config() {
        return config;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void build(List<Shape> inputShapes) {
        Shape inputShape = inputShapes.getFirst();
        Shape projShape = Shape.of(inputShape.dim(0), config.projDim);

        norm1.build(List.of(inputShape));
        norm2.build(List.of(inputShape));
        upProj.build(List.of(inputShape));
        downProj.build(List.of(projShape));
        attention.build(List.of(inputShape));

        if (gateProj != null) {
            gateProj.build(List.of(inputShape));
        }
    }

    /**
     * {@inheritDoc}
     */
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

    /**
     * {@inheritDoc}
     */
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

    /**
     * {@inheritDoc}
     */
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

        x = input.add(x);

        Tensor residual = x;

        x = norm2.forward(cache, x);

        Tensor y = upProj.forward(cache, x);

        if (gateProj != null) {
            x = gateProj.forward(cache, x);
            x = x.activate(activation);
            x = x.mul(y);
        } else {
            x = y.activate(activation);
        }

        x = downProj.forward(cache, x);

        if (cache.isTraining()) {
            x = dropout.forward(cache, x);
        }

        x = residual.add(x);

        return new Tensor[] { x };
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void backward(Updater updater, Optimizer optimizer, int samples) {
        norm2.backward(updater, optimizer, samples);
        downProj.backward(updater, optimizer, samples);
        upProj.backward(updater, optimizer, samples);
        norm1.backward(updater, optimizer, samples);
        attention.backward(updater, optimizer, samples);

        if (gateProj != null) {
            gateProj.backward(updater, optimizer, samples);
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void resetGrad() {
        norm1.resetGrad();
        norm2.resetGrad();
        upProj.resetGrad();
        downProj.resetGrad();
        attention.resetGrad();

        if (gateProj != null) {
            gateProj.resetGrad();
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Layer freeze() {
        norm1.freeze();
        norm2.freeze();
        upProj.freeze();
        downProj.freeze();
        attention.freeze();

        if (gateProj != null) {
            gateProj.freeze();
        }

        return super.freeze();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Layer unfreeze() {
        norm1.unfreeze();
        norm2.unfreeze();
        upProj.unfreeze();
        downProj.unfreeze();
        attention.unfreeze();

        if (gateProj != null) {
            gateProj.unfreeze();
        }

        return super.unfreeze();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public T copy() {
        T copy = copyBlock();

        copy.norm1 = norm1.copy();
        copy.norm2 = norm2.copy();
        copy.upProj = upProj.copy();
        copy.downProj = downProj.copy();
        copy.attention = attention.copy();
        copy.dropout = dropout.copy();

        if (gateProj != null) {
            copy.gateProj = gateProj.copy();
        }

        return copy;
    }

    /**
     * Encoder block with bidirectional attention.
     * Use it to build transformer encoders.
     */
    public static class Encoder extends Transformer<Encoder> {

        /**
         * Create a block with GELU activation.
         */
        public Encoder(int embedDim, int heads, double dropout) {
            super(embedDim, heads, dropout);
        }

        /**
         * Create a block with the given activation.
         */
        public Encoder(int embedDim, int heads, double dropout, Activation activation) {
            super(embedDim, heads, dropout, activation);
        }

        /**
         * Create a block with the given config.
         */
        public Encoder(Config config) {
            super(config);
        }

        /**
         * {@inheritDoc}
         */
        @Override
        protected Encoder copyBlock() {
            return new Encoder(config);
        }

        /**
         * {@inheritDoc}
         */
        @Override
        protected Layer attention() {
            return new MultiHeadAttention(config.heads, config.embedDim);
        }
    }

    /**
     * Decoder block with causal attention.
     * Use it to build autoregressive transformers.
     */
    public static class Decoder extends Transformer<Decoder> {

        /**
         * Create a block with GELU activation.
         */
        public Decoder(int embedDim, int heads, double dropout) {
            super(embedDim, heads, dropout);
        }

        /**
         * Create a block with the given activation.
         */
        public Decoder(int embedDim, int heads, double dropout, Activation activation) {
            super(embedDim, heads, dropout, activation);
        }

        /**
         * Create a block with the given config.
         */
        public Decoder(Config config) {
            super(config);
        }

        /**
         * {@inheritDoc}
         */
        @Override
        protected Decoder copyBlock() {
            return new Decoder(config);
        }

        /**
         * {@inheritDoc}
         */
        @Override
        protected Layer attention() {
            return new MaskedMultiHeadAttention(config.heads, config.embedDim);
        }
    }
}
