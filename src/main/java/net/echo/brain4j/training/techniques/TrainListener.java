package net.echo.brain4j.training.techniques;

import net.echo.brain4j.model.Model;
import net.echo.brain4j.training.data.DataSet;

public class TrainListener {

    protected Model model;

    public void register(Model model) {
        this.model = model;
    }

    /**
     * Called when the trainer completes an epoch
     * @param epoch the current epoch
     */
    public void onEpochCompleted(int epoch) {
    }

    /**
     * Called before the trainer starts an epoch
     * @param epoch
     */
    public void onEpochStarted(int epoch) {
    }

    public void onEvaluated(DataSet dataSet, int epoch, double loss, long took) {
    }

    public void onLossIncreased(double loss, double previousLoss) {
    }
}
