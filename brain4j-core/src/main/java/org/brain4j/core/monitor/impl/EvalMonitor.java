package org.brain4j.core.monitor.impl;

import org.brain4j.core.Brain4J;
import org.brain4j.math.loss.LossFunction;
import org.brain4j.core.model.Model;
import org.brain4j.core.monitor.Monitor;
import org.brain4j.core.training.Trainer;
import org.brain4j.core.training.events.EpochEnd;
import org.brain4j.core.training.events.TrainingEvent;
import org.brain4j.core.training.wrappers.EvaluationResult;
import org.brain4j.core.utils.Colored;
import org.brain4j.math.data.ListDataSource;

public class EvalMonitor implements Monitor {
    
    protected final ListDataSource dataSource;
    protected final int evaluationDelay;
    protected final boolean logging;

    protected EvaluationResult lastRecordedEvaluation;

    public EvalMonitor(ListDataSource dataSource, int evaluationDelay) {
        this(dataSource, evaluationDelay, Brain4J.isLogging());
    }

    public EvalMonitor(ListDataSource dataSource, int evaluationDelay, boolean logging) {
        this.dataSource = dataSource;
        this.evaluationDelay = evaluationDelay;
        this.logging = logging;
        
        if (evaluationDelay <= 0) {
            throw new IllegalArgumentException("evaluationDelay must be greater than 0. Got: " + evaluationDelay);
        }
    }
    
    @Override
    public void onEvent(TrainingEvent event, Trainer trainer) {
        if (event instanceof EpochEnd(int epoch, int total)) {
            if ((epoch + 1) % evaluationDelay != 0) return;

            evaluate(trainer, epoch, total);
        }
    }
    
    protected void evaluate(Trainer trainer, int epoch, int epochs) {
        Model model = trainer.model();
        LossFunction lossFunction = trainer.config().loss();
        EvaluationResult result = model.evaluate(dataSource, lossFunction);

        this.lastRecordedEvaluation = result;

        double r2 = result.loss() / result.totalDeviation();
        boolean regression = lossFunction.isRegression();
        
        double f1 = result.f1Score() * 100.0;
        double accuracy = result.accuracy() * 100.0;
        
        String lossMsg = Colored.renderText("Loss: <magenta>%." + Brain4J.getDecimalDigits() + "f<reset>", result.loss());
        String firstMetric = regression
            ? Colored.renderText(" | R^2 Score: <blue>%.2f<reset>", r2)
            : Colored.renderText(" | Accuracy: <blue>%.2f%%<reset>", accuracy);
        String secondMetric = regression ? "" : Colored.renderText(" | F1-Score: <green>%.2f%%<reset>", f1);
        String prefix = Colored.renderText("Epoch <yellow>%s<white>/<yellow>%s<white> ", epoch + 1, epochs);


        String message = prefix + lossMsg + firstMetric + secondMetric + "\n";
        System.out.print("\n\r" + message);
    }

    public EvaluationResult getEvalResult() {
        return lastRecordedEvaluation;
    }
}
