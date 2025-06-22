package org.brain4j.core.training.advanced;

import org.brain4j.core.model.Model;
import org.brain4j.core.training.evaluation.EvaluationResult;
import org.brain4j.math.data.ListDataSource;

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
    
    public void step(Model model, ListDataSource dataSource, int totalEpoches) {
        long start = System.nanoTime();
        this.listeners.forEach(listener -> listener.onEpochStarted(epoches, totalEpoches, start));
        
        model.fit(dataSource);
        
        long took = System.nanoTime() - start;
        this.listeners.forEach(listener -> listener.onEpochCompleted(epoches, totalEpoches, took));
    }
    
    private void setupPreTraining(Model model, ListDataSource trainSource, ListDataSource evalSource) {
        if (trainSource == null || evalSource == null) {
            throw new IllegalArgumentException("Training source and evaluation source are required and cannot be null!");
        }
        
        this.start = System.nanoTime();
        this.running = true;
        this.epoches = 0;
        
        this.evaluation = model.evaluate(evalSource);
        this.listeners.forEach(listener -> listener.register(this, model));
    }
    
    public void start(Model model, ListDataSource trainSource, ListDataSource evalSource, double lossThreshold) {
        setupPreTraining(model, trainSource, evalSource);
        
        while (running && evaluation.loss() > lossThreshold) {
            iterate(model, trainSource, evalSource, Integer.MAX_VALUE);
        }

        this.running = false;
        this.end = System.nanoTime();
    }
    
    public void startFor(Model model, ListDataSource trainSource, ListDataSource evalSource, int epochesAmount) {
        setupPreTraining(model, trainSource, evalSource);
        
        for (int i = 0; i < epochesAmount && running; i++) {
            iterate(model, trainSource, evalSource, epochesAmount);
        }

        this.running = false;
        this.end = System.nanoTime();
    }

    private void iterate(Model model, ListDataSource trainSource, ListDataSource evalSource, int totalEpoches) {
        step(model, trainSource, totalEpoches);

        this.epoches++;

        if (epoches % evaluateEvery == 0) {
            long start = System.nanoTime();
            this.evaluation = model.evaluate(evalSource);
            long took = System.nanoTime() - start;

            this.listeners.forEach(listener -> listener.onEvaluated(evalSource, evaluation, epoches, took));
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
