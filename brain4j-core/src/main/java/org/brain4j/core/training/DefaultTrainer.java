package org.brain4j.core.training;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.loss.LossFunction;
import org.brain4j.core.model.Model;
import org.brain4j.core.monitor.Monitor;
import org.brain4j.core.training.events.*;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.math.commons.Batch;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.gpu.silicon.SiliconDevice;
import org.brain4j.math.tensor.Tensor;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class DefaultTrainer implements Trainer {
    
    private final Model model;
    private final TrainingConfig config;
    private final Map<Class<? extends Monitor>, Monitor> monitors;
    private final Object trainingLock = new Object();

    private volatile boolean training;
    private volatile boolean paused;
    private volatile boolean stopRequested;
    private volatile int currentEpoch = -1;
    private volatile int currentBatch = -1;
    private volatile int totalEpochs = -1;
    private volatile int totalBatches = -1;
    
    DefaultTrainer(Model model, TrainingConfig config, List<Monitor> monitors) {
        if (model == null) throw new IllegalArgumentException("Model cannot be null!");
        if (config == null) throw new IllegalArgumentException("Config cannot be null!");
        
        config.optimizer().initialize();
        config.updater().initialize();
        
        this.model = model;
        this.config = config;
        this.monitors = new HashMap<>();

        monitors.forEach(m -> this.monitors.put(m.getClass(), m));
    }

    @Override
    public <T extends Monitor> T getMonitor(Class<T> monitorClass) {
        return (T) monitors.get(monitorClass);
    }

    @Override
    public <T extends Monitor> void attach(T monitor) {
        monitors.put(monitor.getClass(), monitor);
    }

    @Override
    public Thread start(ListDataSource dataSource, int epochs) {
        if (dataSource == null) throw new IllegalArgumentException("Data source cannot be null!");
        if (epochs <= 0) throw new IllegalArgumentException("Epochs must be greater than 0. Got: " + epochs);

        synchronized (trainingLock) {
            if (training) {
                throw new IllegalStateException("The model is already being trained!");
            }

            training = true;
            paused = false;
            stopRequested = false;
        }

        Thread thread = new Thread(() -> {
            try {
                runTraining(dataSource, epochs);
            } finally {
                finishTraining();
            }
        }, "brain4j-trainer-thread");

        thread.start();
        return thread;
    }

    @Override
    public void fit(ListDataSource dataSource, int epochs) {
        if (dataSource == null) throw new IllegalArgumentException("Data source cannot be null!");
        if (epochs <= 0) throw new IllegalArgumentException("Epochs must be greater than 0. Got: " + epochs);

        synchronized (trainingLock) {
            if (training) {
                throw new IllegalStateException("The model is already being trained!");
            }

            training = true;
            paused = false;
            stopRequested = false;
        }

        try {
            runTraining(dataSource, epochs);
        } finally {
            finishTraining();
        }
    }

    @Override
    public void pause() {
        synchronized (trainingLock) {
            if (!training) {
                throw new IllegalStateException("The model is not being trained!");
            }

            paused = true;
        }
    }

    @Override
    public void resume() {
        synchronized (trainingLock) {
            if (!training) {
                throw new IllegalStateException("The model is not being trained!");
            }

            if (!paused) return;

            paused = false;
            trainingLock.notifyAll();
        }
    }

    @Override
    public void stop() {
        synchronized (trainingLock) {
            if (!training) return;

            stopRequested = true;
            paused = false;
            trainingLock.notifyAll();
        }
    }

    @Override
    public boolean isTraining() {
        return training;
    }

    @Override
    public boolean isPaused() {
        return paused;
    }

    @Override
    public int currentEpoch() {
        return currentEpoch;
    }

    @Override
    public int currentBatch() {
        return currentBatch;
    }

    @Override
    public int totalEpochs() {
        return totalEpochs;
    }

    @Override
    public int totalBatches() {
        return totalBatches;
    }

    private void runTraining(ListDataSource dataSource, int epochs) {
        totalEpochs = epochs;

        for (int i = 0; i < epochs; i++) {
            if (shouldStop()) break;
            fitEpoch(dataSource, i, epochs);
        }
    }

    private void finishTraining() {
        synchronized (trainingLock) {
            training = false;
            paused = false;
            stopRequested = false;
            currentEpoch = -1;
            currentBatch = -1;
            totalEpochs = -1;
            totalBatches = -1;
            trainingLock.notifyAll();
        }

        monitors.forEach((k, x) -> x.onEvent(new TrainingEnd(), this));
    }
    
    private void fitEpoch(ListDataSource dataSource, int index, int total) {
        currentEpoch = index;
        currentBatch = -1;

        EpochStart epochStart = new EpochStart(index, total);
        monitors.forEach((k, x) -> x.onEvent(epochStart, this));
        
        dataSource.reset();
        int totalBatches = dataSource.getBatches();
        this.totalBatches = totalBatches;
        
        while (dataSource.hasNext()) {
            if (shouldStop()) break;

            int cursor = dataSource.getCursor();
            currentBatch = cursor;
            
            Batch batch = dataSource.nextBatch().to(model.getDevice());
            
            BatchStart batchStart = new BatchStart(this, cursor, totalBatches);
            monitors.forEach((k, x) -> x.onEvent(batchStart, this));
            
            fitBatch(batch);
            
            BatchEnd end = new BatchEnd(cursor, totalBatches);
            monitors.forEach((k, x) -> x.onEvent(end, this));
        }
        
        Optimizer optimizer = config.optimizer();
        Updater updater = config.updater();
        
        updater.postFit(optimizer.getLearningRate(), dataSource.getSize());
        
        EpochEnd end = new EpochEnd(index, total);
        monitors.forEach((k, x) -> x.onEvent(end, this));
    }

    private boolean shouldStop() {
        synchronized (trainingLock) {
            while (paused && !stopRequested) {
                try {
                    trainingLock.wait();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    stopRequested = true;
                    paused = false;
                    break;
                }
            }

            return stopRequested;
        }
    }
    
    private void fitBatch(Batch batch) {
        Tensor[] inputs = batch.getFirst();
        
        SiliconDevice device = model.getDevice();
        StatesCache cache = new StatesCache(true);
        
        if (device != null) {
            device.createResources();
        }
        
        Tensor[] outputs = forward(cache, inputs);
        backward(cache, batch, outputs);
        resetGrad();
        
        if (device != null) {
            device.closeResources();
        }
    }
    
    @Override
    public Tensor[] forward(StatesCache cache, Tensor[] inputs) {
        return model.predict(cache, inputs);
    }
    
    @Override
    public void backward(StatesCache cache, Batch batch, Tensor[] outputs) {
        List<Layer> layers = model.getLayers();
        
        Tensor[] inputs = batch.getFirst();
        Tensor[] targets = batch.getSecond();
        
        Updater updater = config.updater();
        Optimizer optimizer = config.optimizer();
        LossFunction loss = config.loss();
        
        // TODO: fix this
//        layers.getLast().computeLoss(cache, targets, outputs, loss);
//        layers.forEach(layer -> layer.backward(cache, updater, optimizer));
        
        int elements = 0;
        
        for (Tensor input : inputs) {
            elements += input.shapeAt(0);
        }
        
        optimizer.postBatch();
        updater.postBatch(optimizer.getLearningRate(), elements);
    }
    
    @Override
    public void resetGrad() {
        model.getLayers().forEach(Layer::resetGrad);
    }
    
    @Override
    public Model model() {
        return model;
    }
    
    @Override
    public TrainingConfig config() {
        return config;
    }
    
    @Override
    public List<Monitor> monitors() {
        return monitors.values().stream().toList();
    }
}
