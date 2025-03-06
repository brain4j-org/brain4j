package net.echo.brain4j.model.impl;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.layer.impl.DropoutLayer;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.initialization.WeightInit;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import net.echo.brain4j.training.updater.impl.StochasticUpdater;
import net.echo.brain4j.transformers.TransformerEncoder;
import net.echo.brain4j.utils.DataSet;
import net.echo.brain4j.utils.MLUtils;
import net.echo.brain4j.utils.Vector;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

public class Transformer extends Model<Object, List<Vector>, List<Vector>> {

    @SafeVarargs
    public Transformer(Layer<List<Vector>, List<Vector>>... layers) {
        super(layers);
    }

    @Override
    public Transformer compile(LossFunctions function, Optimizer optimizer) {
        return compile(WeightInit.UNIFORM_XAVIER, function, optimizer, new StochasticUpdater());
    }

    @Override
    public Transformer compile(WeightInit weightInit, LossFunctions function, Optimizer optimizer, Updater updater) {
        super.compile(weightInit, function, optimizer, updater);

        connect(weightInit, true);

        return this;
    }

    @Override
    public void connect(WeightInit weightInit, boolean update) {
        super.connect(weightInit, update);
    }

    @Override
    public double evaluate(DataSet<Object> set) {
        return 0;
    }

    @Override
    public void fit(DataSet<Object> dataSet) {

    }

    @Override
    public List<Vector> predict(StatesCache cache, List<Vector> input) {
        List<Vector> result = new ArrayList<>(input);

        for (Layer<?, ?> layer : layers) {
            if (layer instanceof TransformerEncoder encoder) {
                result = encoder.forward(cache, layer, input);
            }
        }

        return result;
    }

    @Override
    public void reloadMatrices() {

    }

    @Override
    public List<Vector> predict(List<Vector> input) {
        return predict(null, input);
    }
}
