package net.echo.brain4j.training.techniques;

import net.echo.brain4j.model.Model;
import net.echo.brain4j.utils.DataSet;

import java.util.ArrayList;
import java.util.List;

public class SmartTrainer<R> {

    private final List<TrainListener<R>> listeners;
    private final double learningRateDecay;
    private final int evaluateEvery;

    private double previousLoss = Double.MAX_VALUE;
    private double loss = Double.MAX_VALUE;
    private boolean running;
    private int epoches;

    private long start;
    private long end;

    public SmartTrainer(double learningRateDecay, int evaluateEvery) {
        this.listeners = new ArrayList<>();
        this.learningRateDecay = learningRateDecay;
        this.evaluateEvery = evaluateEvery;
    }

    public void addListener(TrainListener<R> listener) {
        listeners.add(listener);
    }

    public void abort() {
        this.running = false;
    }

    public void start(Model<R, ?, ?> model, DataSet<R> dataSet, double lossThreshold, double lossTolerance) {
        this.start = System.nanoTime();
        this.running = true;
        this.epoches = 0;

        this.listeners.forEach(listener -> listener.register(this, model));

        while (running && loss > lossThreshold) {
            step(model, dataSet);

            this.epoches++;

            if (epoches % evaluateEvery == 0) {
                long start = System.nanoTime();
                this.loss = model.loss(dataSet);
                long took = System.nanoTime() - start;

                this.listeners.forEach(listener -> listener.onEvaluated(dataSet, epoches, loss, took));

                if ((loss - previousLoss) > lossTolerance) {
                    // Loss increased, so decrease the learning rate
                    model.getOptimizer().setLearningRate(model.getOptimizer().getLearningRate() * learningRateDecay);
                    this.listeners.forEach(listener -> listener.onLossIncreased(loss, previousLoss));
                }

                previousLoss = loss;
            }
        }

        this.running = false;
        this.end = System.nanoTime();
    }

    public void step(Model<R, ?, ?> model, DataSet<R> dataSet) {
        long start = System.nanoTime();
        this.listeners.forEach(listener -> listener.onEpochStarted(epoches, start));

        model.fit(dataSet);

        long took = System.nanoTime() - start;
        this.listeners.forEach(listener -> listener.onEpochCompleted(epoches, took));
    }

    public void startFor(Model<R, ?, ?> model, DataSet<R> dataSet, int epochesAmount, double lossTolerance) {
        this.start = System.nanoTime();
        this.running = true;
        this.epoches = 0;

        this.listeners.forEach(listener -> listener.register(this, model));

        for (int i = 0; i < epochesAmount && running; i++) {
            step(model, dataSet);

            this.epoches++;

            if (epoches % evaluateEvery == 0) {
                long start = System.nanoTime();
                this.loss = model.loss(dataSet);
                long took = System.nanoTime() - start;

                this.listeners.forEach(listener -> listener.onEvaluated(dataSet, epoches, loss, took));

                if ((loss - previousLoss) > lossTolerance) {
                    // Loss increased, so decrease the learning rate
                    model.getOptimizer().setLearningRate(model.getOptimizer().getLearningRate() * learningRateDecay);
                    this.listeners.forEach(listener -> listener.onLossIncreased(loss, previousLoss));
                }

                previousLoss = loss;
            }
        }

        this.running = false;
        this.end = System.nanoTime();
    }

    public int getEpoches() {
        return epoches;
    }

    public boolean isRunning() {
        return running;
    }

    public double getLoss() {
        return loss;
    }

    public double getPreviousLoss() {
        return previousLoss;
    }

    public double getLearningRateDecay() {
        return learningRateDecay;
    }

    public int getEvaluateEvery() {
        return evaluateEvery;
    }

    public List<TrainListener<R>> getListeners() {
        return listeners;
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
}
