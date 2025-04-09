package net.echo.brain4j.training.advanced;

import net.echo.brain4j.model.Model;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.evaluation.EvaluationResult;
import net.echo.math4j.DataSet;

import java.util.ArrayList;
import java.util.List;

public class SmartTrainer {

    private final List<TrainListener> listeners;
    private final double learningRateDecay;
    private final int evaluateEvery;

    private EvaluationResult evaluation;
    private boolean running;
    private int epoches;

    private long start;
    private long end;

    public SmartTrainer(double learningRateDecay, int evaluateEvery) {
        this.listeners = new ArrayList<>();
        this.learningRateDecay = learningRateDecay;
        this.evaluateEvery = evaluateEvery;
    }

    public void addListener(TrainListener listener) {
        listeners.add(listener);
    }

    public void abort() {
        this.running = false;
    }

    public void start(Model model, DataSet<DataRow> dataSet, double lossThreshold) {
        this.start = System.nanoTime();
        this.running = true;
        this.epoches = 0;

        this.listeners.forEach(listener -> listener.register(this, model));

        while (running && evaluation.loss() > lossThreshold) {
            iterate(model, dataSet, Integer.MAX_VALUE);
        }

        this.running = false;
        this.end = System.nanoTime();
    }

    public void step(Model model, DataSet<DataRow> dataSet, int totalEpoches) {
        long start = System.nanoTime();
        this.listeners.forEach(listener -> listener.onEpochStarted(epoches, totalEpoches, start));

        model.fit(dataSet);

        long took = System.nanoTime() - start;
        this.listeners.forEach(listener -> listener.onEpochCompleted(epoches, totalEpoches, took));
    }

    public void startFor(Model model, DataSet<DataRow> dataSet, int epochesAmount) {
        this.start = System.nanoTime();
        this.running = true;
        this.epoches = 0;

        this.listeners.forEach(listener -> listener.register(this, model));

        for (int i = 0; i < epochesAmount && running; i++) {
            iterate(model, dataSet, epochesAmount);
        }

        this.running = false;
        this.end = System.nanoTime();
    }

    private void iterate(Model model, DataSet<DataRow> dataSet, int totalEpoches) {
        step(model, dataSet, totalEpoches);

        this.epoches++;

        if (epoches % evaluateEvery == 0) {
            long start = System.nanoTime();
            EvaluationResult result = model.evaluate(dataSet);
            long took = System.nanoTime() - start;

            this.listeners.forEach(listener -> listener.onEvaluated(dataSet, result, epoches, took));
        }
    }

    public int getEpoches() {
        return epoches;
    }

    public boolean isRunning() {
        return running;
    }

    public EvaluationResult getEvaluation() {
        return evaluation;
    }

    public double getLearningRateDecay() {
        return learningRateDecay;
    }

    public int getEvaluateEvery() {
        return evaluateEvery;
    }

    public long getStart() {
        return start;
    }

    public long getEnd() {
        return end;
    }

    public long getTook() {
        return end - start;
    }

    public List<TrainListener> getListeners() {
        return listeners;
    }
}
