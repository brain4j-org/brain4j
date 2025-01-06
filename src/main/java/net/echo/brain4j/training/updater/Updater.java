package net.echo.brain4j.training.updater;

import com.google.gson.annotations.JsonAdapter;
import net.echo.brain4j.adapters.OptimizerAdapter;
import net.echo.brain4j.adapters.UpdaterAdapter;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.Synapse;

import java.util.List;

@JsonAdapter(UpdaterAdapter.class)
public abstract class Updater {

    public void postInitialize() {
    }

    public void postIteration(List<Layer> layers, double learningRate) {
    }

    public void postFit(List<Layer> layers, double learningRate) {
    }

    public void postBatch(List<Layer> layers, double learningRate) {
    }

    public abstract void acknowledgeChange(Synapse synapse, double change);
}
