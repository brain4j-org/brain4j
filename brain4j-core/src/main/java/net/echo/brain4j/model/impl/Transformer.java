package net.echo.brain4j.model.impl;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.initialization.WeightInit;
import net.echo.brain4j.model.initialization.WeightInitializer;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.brain4j.training.evaluation.EvaluationResult;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import net.echo.brain4j.training.updater.impl.StochasticUpdater;
import net.echo.brain4j.transformers.TransformerDecoder;
import net.echo.brain4j.transformers.TransformerEncoder;
import net.echo.math4j.DataSet;
import net.echo.math4j.math.tensor.Tensor;

import java.io.DataInputStream;
import java.io.DataOutputStream;

public class Transformer extends Model<Object, Tensor, Tensor> {

    @SafeVarargs
    public Transformer(Layer<Tensor, Tensor>... layers) {
        super(layers);
    }

    @Override
    public Transformer compile(LossFunction function, Optimizer optimizer) {
        return compile(WeightInit.UNIFORM_XAVIER.getFunction(), function, optimizer, new StochasticUpdater());
    }

    @Override
    public Transformer compile(LossFunctions function, Optimizer optimizer) {
        return compile(function.getFunction(), optimizer);
    }

    @Override
    public Transformer compile(WeightInit initializer, LossFunctions lossFunction, Optimizer optimizer, Updater updater) {
        return compile(initializer.getFunction(), lossFunction.getFunction(), optimizer, updater);
    }

    @Override
    public Transformer compile(WeightInitializer initializer, LossFunction lossFunction, Optimizer optimizer, Updater updater) {
        super.compile(initializer, lossFunction, optimizer, updater);

        connect(initializer, true);

        return this;
    }

    @Override
    public EvaluationResult evaluate(DataSet<Object> dataSet) {
        return null;
    }

    @Override
    public void connect(WeightInitializer weightInit, boolean update) {
        super.connect(weightInit, update);
    }

    @Override
    public double loss(DataSet<Object> dataSet) {
        return 0;
    }

    @Override
    public void fit(DataSet<Object> dataSet) {
        throw new UnsupportedOperationException("Not implemented yet for this class.");
    }

    @Override
    public Tensor predict(StatesCache cache, Tensor input, boolean training) {
        Tensor result = input;

        for (Layer<?, ?> layer : layers) {
            if (layer instanceof TransformerEncoder encoder) {
                result = encoder.forward(cache, layer, result);
            }

            if (layer instanceof TransformerDecoder decoder) {
                result = decoder.forward(cache, layer, result);
            }
        }

        return result;
    }

    @Override
    public void reloadWeights() {

    }

    @Override
    public void serialize(DataOutputStream stream) throws Exception {
        throw new UnsupportedOperationException("Not implemented yet for this class.");
    }

    @Override
    public void deserialize(DataInputStream stream) throws Exception {
        throw new UnsupportedOperationException("Not implemented yet for this class.");
    }
}
