package org.brain4j.core.training;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.model.Model;
import org.brain4j.core.monitor.Monitor;
import org.brain4j.core.training.events.BatchEnd;
import org.brain4j.core.training.events.BatchStart;
import org.brain4j.core.training.events.EpochEnd;
import org.brain4j.core.training.events.EpochStart;
import org.brain4j.core.training.events.TrainingEnd;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.core.utils.Colored;
import org.brain4j.core.utils.ProgressBar;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.impl.Softmax;
import org.brain4j.math.commons.Batch;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.loss.LossFunction;
import org.brain4j.math.loss.impl.CrossEntropy;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.AutogradContext;
import org.brain4j.math.tensor.autograd.Operation;
import org.brain4j.math.tensor.autograd.impl.ActivationOperation;

/**
 * Runs epoch and batch loops with monitors and progress reporting.
 * Use it as the default synchronous and background trainer.
 */
@SuppressWarnings("unchecked")
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
        Objects.requireNonNull(model, "Model cannot be null!");
        Objects.requireNonNull(config, "Config cannot be null!");

        config.optimizer().initialize();
        config.updater().initialize();

        this.model = model;
        this.config = config;
        this.monitors = new HashMap<>();

        monitors.forEach(m -> this.monitors.put(m.getClass(), m));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends Monitor> T getMonitor(Class<T> monitorClass) {
        return (T) monitors.get(monitorClass);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends Monitor> void attach(T monitor) {
        monitors.put(monitor.getClass(), monitor);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Thread start(ListDataSource dataSource, int epochs) {
        checkAndSync(dataSource, epochs);

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

    /**
     * {@inheritDoc}
     */
    @Override
    public void fit(ListDataSource dataSource, int epochs) {
        checkAndSync(dataSource, epochs);

        try {
            runTraining(dataSource, epochs);
        } finally {
            finishTraining();
        }
    }

    private void checkAndSync(ListDataSource dataSource, int epochs) {
        Objects.requireNonNull(dataSource, "Data source cannot be null!");

        if (epochs <= 0) {
            throw new IllegalArgumentException("Epochs must be greater than 0. Got: " + epochs);
        }

        synchronized (trainingLock) {
            if (training) {
                throw new IllegalStateException("The model is already being trained!");
            }

            training = true;
            paused = false;
            stopRequested = false;
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void pause() {
        synchronized (trainingLock) {
            if (!training) {
                throw new IllegalStateException("The model is not being trained!");
            }

            paused = true;
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void resume() {
        synchronized (trainingLock) {
            if (!training) {
                throw new IllegalStateException("The model is not being trained!");
            }

            if (!paused) {
                return;
            }

            paused = false;
            trainingLock.notifyAll();
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void stop() {
        synchronized (trainingLock) {
            if (!training) {
                return;
            }

            stopRequested = true;
            paused = false;
            trainingLock.notifyAll();
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean isTraining() {
        return training;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean isPaused() {
        return paused;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int currentEpoch() {
        return currentEpoch;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int currentBatch() {
        return currentBatch;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int totalEpochs() {
        return totalEpochs;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int totalBatches() {
        return totalBatches;
    }

    private void runTraining(ListDataSource dataSource, int epochs) {
        totalEpochs = epochs;
        totalBatches = dataSource.batches();

        for (int i = 0; i < epochs; i++) {
            if (shouldStop()) {
                break;
            }

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
        currentBatch = 0;

        EpochStart epochStart = new EpochStart(index, total);
        monitors.forEach((k, x) -> x.onEvent(epochStart, this));

        Iterable<Batch> batchIterator = dataSource.batchIterator();
        String taskName = Colored.renderText("Epoch <yellow>%s<white>/<yellow>%s ", index, total);

        for (Batch batch : new ProgressBar<>(batchIterator, taskName)) {
            if (shouldStop()) {
                break;
            }

            currentBatch++;

            BatchStart batchStart = new BatchStart(this, currentBatch, totalBatches);
            monitors.forEach((k, x) -> x.onEvent(batchStart, this));

            fitBatch(batch.to(model.device()));

            BatchEnd end = new BatchEnd(currentBatch, totalBatches);
            monitors.forEach((k, x) -> x.onEvent(end, this));
        }

        Updater updater = config.updater();

        updater.postFit();

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
        Tensor[] inputs = batch.first();

        Device device = model.device();
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

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor[] forward(StatesCache cache, Tensor[] inputs) {
        return model.predict(cache, inputs);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void backward(StatesCache cache, Batch batch, Tensor[] outputs) {
        List<Layer> layers = model.layers();

        Tensor[] inputs = batch.first();
        Tensor[] targets = batch.second();

        if (inputs.length == 0) {
            throw new IllegalArgumentException("Batch must contain at least one input.");
        }

        int batchSize = inputs[0].shapeAt(0);

        if (batchSize <= 0) {
            throw new IllegalArgumentException("Batch size must be greater than 0.");
        }

        for (int i = 1; i < inputs.length; i++) {
            if (inputs[i].shapeAt(0) != batchSize) {
                throw new IllegalArgumentException("All inputs must have the same batch size.");
            }
        }

        Updater updater = config.updater();
        Optimizer optimizer = config.optimizer();
        LossFunction loss = config.loss();

        for (int i = 0; i < outputs.length; i++) {
            Tensor y = outputs[i];
            Tensor t = targets[i];

            AutogradContext context = y.autogradContext();
            Operation operation = context.operation();

            Tensor grad = loss.delta(y, t, null);
            Tensor target = y;

            if (loss instanceof CrossEntropy
                && operation instanceof ActivationOperation(Activation activation)
                && activation instanceof Softmax) {
                // Use pre-softmax logits for a stable combined gradient.
                target = context.inputs()[0];
            }

            target.backward(grad);
        }

        layers.forEach(layer -> layer.backward(updater, optimizer, batchSize));

        optimizer.postBatch();
        updater.postBatch();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void resetGrad() {
        model.layers().forEach(Layer::resetGrad);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Model model() {
        return model;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public TrainingConfig config() {
        return config;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public List<Monitor> monitors() {
        return monitors.values().stream().toList();
    }
}
