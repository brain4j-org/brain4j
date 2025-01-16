package net.echo.brain4j.training.techniques;

import net.echo.brain4j.model.Model;
import net.echo.brain4j.training.data.DataSet;

public class TrainListener {

    protected Model model;

    public void register(Model model) {
        this.model = model;
    }

    public void onEpochCompleted(int epoch) {
    }

    public void onEpochStarted(int epoch) {
    }

    public void onEvaluated(DataSet dataSet, int epoch, double loss) {
    }

    public void onLossIncreased(double loss, double previousLoss) {
    }
}
