package net.echo.brain4j.training.optimizer;

import com.google.gson.annotations.JsonAdapter;
import net.echo.brain4j.adapters.Adapter;
import net.echo.brain4j.adapters.json.OptimizerAdapter;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.structure.StatesCache;
import net.echo.brain4j.training.updater.Updater;
import net.echo.math4j.math.tensor.Tensor;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.List;

@JsonAdapter(OptimizerAdapter.class)
public abstract class Optimizer implements Adapter {

    protected double learningRate;

    protected Optimizer() {
    }

    public Optimizer(double learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public void serialize(DataOutputStream stream) throws Exception {
        stream.writeDouble(learningRate);
    }

    @Override
    public void deserialize(DataInputStream stream) throws Exception {
        this.learningRate = stream.readDouble();
    }

    public abstract Tensor optimize(Layer layer, Tensor delta, Tensor output);

    public void postInitialize(Model model) {
    }

    public void postIteration(StatesCache cacheHolder, Updater updater, List<Layer> layers) {
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
}