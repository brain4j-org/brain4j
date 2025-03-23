package net.echo.brain4j.training.techniques;

import net.echo.brain4j.model.Model;
import net.echo.math4j.DataSet;

public class TrainListener<R> {

    protected SmartTrainer<R> trainer;
    protected Model<?, ?, ?> model;

    public void register(SmartTrainer<R> trainer, Model<?, ?, ?> model) {
        this.trainer = trainer;
        this.model = model;
    }

    /**
     * Called when the trainer completes an epoch
     * @param epoch the epoch that has completed
     */
    public void onEpochCompleted(int epoch, long took) {
    }

    /**
     * Called before the trainer starts an epoch
     * @param epoch the current epoch that has started
     */
    public void onEpochStarted(int epoch, long start) {
    }

    public void onEvaluated(DataSet<R> dataSet, int epoch, double loss, long took) {
    }

    public void onLossIncreased(double loss, double previousLoss) {
    }
}
