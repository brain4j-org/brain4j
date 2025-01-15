package net.echo.brain4j.training.updater;

import com.google.gson.annotations.JsonAdapter;
import net.echo.brain4j.adapters.UpdaterAdapter;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.structure.Synapse;

import java.util.List;

@JsonAdapter(UpdaterAdapter.class)
public abstract class Updater {

    public void postInitialize(Model model) {
    }

    public void postIteration(Model model, double learningRate) {
    }

    public void postFit(Model model, double learningRate) {
    }

    public void postBatch(Model model, double learningRate) {
    }

    public abstract void acknowledgeChange(Synapse synapse, double change);
}
