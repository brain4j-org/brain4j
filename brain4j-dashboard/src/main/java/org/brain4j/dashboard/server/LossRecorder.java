package org.brain4j.dashboard.server;

import org.brain4j.core.monitor.Monitor;
import org.brain4j.core.monitor.impl.EvalMonitor;
import org.brain4j.core.training.Trainer;
import org.brain4j.core.training.events.EpochEnd;
import org.brain4j.core.training.events.TrainingEvent;
import org.brain4j.core.training.wrappers.EvaluationResult;

import java.util.HashMap;
import java.util.Map;

public class LossRecorder implements Monitor {

    private final EvalMonitor evalMonitor;
    private final Map<Integer, Double> recordedLoss;
    private final Map<Integer, Double> recordedAccuracy;
    private final Map<Integer, Double> recordedF1;

    public LossRecorder(EvalMonitor evalMonitor) {
        this.evalMonitor = evalMonitor;
        this.recordedLoss = new HashMap<>();
        this.recordedAccuracy = new HashMap<>();
        this.recordedF1 = new HashMap<>();
    }

    @Override
    public void onEvent(TrainingEvent event, Trainer trainer) {
        if (event instanceof EpochEnd(int epoch, int total)) {
            EvaluationResult result = evalMonitor.getEvalResult();

            if (result == null) return;

            recordedLoss.put(epoch, result.loss());
            recordedAccuracy.put(epoch, result.accuracy());
            recordedF1.put(epoch, result.f1Score());
        }
    }

    public Map<Integer, Double> getRecordedLoss() {
        return recordedLoss;
    }

    public Map<Integer, Double> getRecordedAccuracy() {
        return recordedAccuracy;
    }

    public Map<Integer, Double> getRecordedF1() {
        return recordedF1;
    }
}
