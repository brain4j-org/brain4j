package net.echo.brain4j.model.impl;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.initialization.WeightInit;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import net.echo.brain4j.transformers.TransformerEncoder;
import net.echo.brain4j.utils.DataSet;
import net.echo.brain4j.utils.Vector;

import java.util.ArrayList;
import java.util.List;

public class Transformer extends Model<Object, List<Vector>, List<Vector>> {

    @SafeVarargs
    public Transformer(Layer<List<Vector>, List<Vector>>... layers) {
        super(layers);
    }

    @Override
    public Model<Object, List<Vector>, List<Vector>> compile(WeightInit weightInit, LossFunctions function, Optimizer optimizer, Updater updater) {
        return super.compile(weightInit, function, optimizer, updater);
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
