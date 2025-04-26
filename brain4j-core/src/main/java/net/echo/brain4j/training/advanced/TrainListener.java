package net.echo.brain4j.training.advanced;

import net.echo.brain4j.model.Model;
import net.echo.brain4j.training.evaluation.EvaluationResult;
import net.echo.math.data.ListDataSource;

public class TrainListener {

    protected SmartTrainer trainer;
    protected Model model;

    public void register(SmartTrainer trainer, Model model) {
        this.trainer = trainer;
        this.model = model;
    }

    /**
     * Called when the trainer completes an epoch
     * @param epoch the epoch that has completed
     */
    public void onEpochCompleted(int epoch, int totalEpoches, long took) {
    }

    /**
     * Called before the trainer starts an epoch
     * @param epoch the current epoch that has started
     */
    public void onEpochStarted(int epoch, int totalEpoches, long start) {
    }

    public void onEvaluated(ListDataSource dataSource, EvaluationResult evaluation, int epoch, long took) {
    }

    public void onLossIncreased(double loss, double previousLoss) {
    }
}
