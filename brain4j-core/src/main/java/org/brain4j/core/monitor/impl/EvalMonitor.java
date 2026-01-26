package org.brain4j.core.monitor.impl;

import org.brain4j.core.Brain4J;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.model.Model;
import org.brain4j.core.monitor.Monitor;
import org.brain4j.core.training.Trainer;
import org.brain4j.core.training.events.EpochEnd;
import org.brain4j.core.training.events.TrainingEvent;
import org.brain4j.core.training.wrappers.EvaluationResult;
import org.brain4j.core.utils.Colored;
import org.brain4j.math.data.ListDataSource;

public class EvalMonitor implements Monitor {
    
    private final ListDataSource dataSource;
    private final int evaluationDelay;
    
    public EvalMonitor(ListDataSource dataSource, int evaluationDelay) {
        this.dataSource = dataSource;
        this.evaluationDelay = evaluationDelay;
    }
    
    @Override
    public void onEvent(TrainingEvent event) {
        if (event instanceof EpochEnd(Trainer trainer, int epoch, int total)) {
            if ((epoch + 1) % evaluationDelay != 0) return;
            
            printEvaluation(trainer, epoch, total);
        }
    }
    
    private void printEvaluation(Trainer trainer, int epoch, int epochs) {
        Model model = trainer.model();
        LossFunction lossFunction = trainer.config().loss();
        EvaluationResult result = model.evaluate(dataSource, lossFunction);
        
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
}
