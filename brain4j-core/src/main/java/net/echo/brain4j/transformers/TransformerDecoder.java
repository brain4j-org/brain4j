package net.echo.brain4j.transformers;

import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.model.initialization.WeightInitializer;
import net.echo.brain4j.training.optimizer.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import net.echo.brain4j.transformers.attention.MaskedMultiHeadAttention;
import net.echo.brain4j.transformers.attention.MultiHeadAttention;

public class TransformerDecoder extends TransformerEncoder {

    private MaskedMultiHeadAttention maskedAttention;

    public TransformerDecoder(int numHeads, int dimension) {
        super(numHeads, dimension);
    }

    @Override
    public void compile(WeightInitializer weightInit, LossFunction lossFunction, Optimizer optimizer, Updater updater) {
        this.maskedAttention = new MaskedMultiHeadAttention(weightInit, heads, dimension);
        this.feedForward.compile(weightInit, lossFunction, optimizer, updater);
    }

    @Override
    public MultiHeadAttention getAttention() {
        return maskedAttention;
    }
}
