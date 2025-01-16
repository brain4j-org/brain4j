package net.echo.brain4j.training.techniques;

import net.echo.brain4j.model.Model;
import net.echo.brain4j.training.data.DataSet;

import java.util.ArrayList;
import java.util.List;

public class SmartTrainer {

    private final List<TrainListener> listeners = new ArrayList<>();

    private final double learningRateDecay;
    private final int evaluateEvery;

    private int epoches;
    private boolean running;

    public SmartTrainer(double learningRateDecay, int evaluateEvery) {
        this.learningRateDecay = learningRateDecay;
        this.evaluateEvery = evaluateEvery;
    }

    public void addListener(TrainListener listener) {
        listeners.add(listener);
    }

    public void abort() {
        this.running = false;
    }

    public void start(Model model, DataSet dataSet, double lossThreshold) {
        this.running = true;
        this.epoches = 0;

        this.listeners.forEach(listener -> listener.register(model));

        double loss = Double.MAX_VALUE;
        double lastLoss = Double.MAX_VALUE;

        while (running && loss > lossThreshold) {
            step(model, dataSet);

            if (epoches++ % evaluateEvery == 0) {
                loss = model.evaluate(dataSet);

                double finalLoss = loss;
                this.listeners.forEach(listener -> listener.onEvaluated(dataSet, epoches, finalLoss));

                if (loss >= lastLoss) {
                    // Loss increased, so decrease the learning rate
                    model.getOptimizer().setLearningRate(model.getOptimizer().getLearningRate() * learningRateDecay);
                }

                lastLoss = loss;
            }
        }

        this.running = false;
    }

    public void step(Model model, DataSet dataSet) {
        this.listeners.forEach(listener -> listener.onEpochStarted(epoches));
        model.fit(dataSet);
        this.listeners.forEach(listener -> listener.onEpochCompleted(epoches));
    }

    public void startFor(Model model, DataSet dataSet, int epochesAmount) {
        this.running = true;
        this.epoches = 0;

        this.listeners.forEach(listener -> listener.register(model));

        double loss;
        double lastLoss = Double.MAX_VALUE;

        for (int i = 0; i < epochesAmount; i++) {
            step(model, dataSet);

            if (epoches++ % evaluateEvery == 0) {
                loss = model.evaluate(dataSet);

                double finalLoss = loss;
                this.listeners.forEach(listener -> listener.onEvaluated(dataSet, epoches, finalLoss));

                if (loss >= lastLoss) {
                    double finalLastLoss = lastLoss;

                    // Loss increased, so decrease the learning rate
                    model.getOptimizer().setLearningRate(model.getOptimizer().getLearningRate() * learningRateDecay);
                    this.listeners.forEach(listener -> listener.onLossIncreased(finalLoss, finalLastLoss));
                }

                lastLoss = loss;
            }
        }

        this.running = false;
    }

    public int getEpoches() {
        return epoches;
    }
}
