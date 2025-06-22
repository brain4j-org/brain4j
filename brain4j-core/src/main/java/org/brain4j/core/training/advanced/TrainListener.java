package org.brain4j.core.training.advanced;

import org.brain4j.core.model.Model;
import org.brain4j.core.training.evaluation.EvaluationResult;
import org.brain4j.math.data.ListDataSource;

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
    
    /**
     * Called after an evaluation has completed.
     * @param dataSource the data source where the evaluation has been done
     * @param evaluation the evaluation result
     * @param epoch the current epoch
     * @param took how many nanoseconds it took to evaluate
     */
    public void onEvaluated(ListDataSource dataSource, EvaluationResult evaluation, int epoch, long took) {
    }
    
    /**
     * Called when the loss gets increased
     * @param loss the current loss
     * @param previousLoss the previous loss
     */
    public void onLossIncreased(double loss, double previousLoss) {
    }
    
    /**
     * Aborts the training session, delegates to {@link SmartTrainer#abort()}
     */
    public void abort() {
        trainer.abort();
    }
}
