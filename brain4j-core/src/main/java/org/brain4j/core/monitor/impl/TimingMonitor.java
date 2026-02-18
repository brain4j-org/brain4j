package org.brain4j.core.monitor.impl;

import org.brain4j.core.monitor.Monitor;
import org.brain4j.core.training.Trainer;
import org.brain4j.core.training.events.*;

import java.util.ArrayList;
import java.util.List;

public class TimingMonitor implements Monitor {

    protected final List<Double> times = new ArrayList<>();
    protected final int timeWindow;

    protected double batchStartTime;
    protected int currentEpoch;
    protected int totalEpochs;
    protected double averageTimeBatch;

    public TimingMonitor(int timeWindow) {
        this.timeWindow = timeWindow;
    }

    @Override
    public void onEvent(TrainingEvent event, Trainer trainer) {
        switch (event) {
            case BatchStart ignored -> this.batchStartTime = System.nanoTime();
            case BatchEnd(int batch, int totalBatches) -> batchCompleted();
            case EpochStart(int epoch, int totalEpochs) -> epochStarted(epoch, totalEpochs);
            default -> {}
        }
    }


    private void batchCompleted() {
        double took = System.nanoTime() - batchStartTime;
        times.add(took / 1e6);

        if (times.size() > timeWindow) {
            times.removeFirst();
        }

        this.averageTimeBatch = times.stream().reduce(Double::sum).orElse(0.0);
    }

    public void epochStarted(int epoch, int total) {
        this.currentEpoch = epoch;
        this.totalEpochs = total;
    }

    public double averagePerBatch() {
        return totalTime() / times.size();
    }

    public double totalTime() {
        return averageTimeBatch;
    }

    public int getCurrentEpoch() {
        return currentEpoch;
    }

    public int getTotalEpochs() {
        return totalEpochs;
    }
}
