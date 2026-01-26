package org.brain4j.core.training;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.model.Model;
import org.brain4j.core.monitor.Monitor;
import org.brain4j.core.training.events.*;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.math.commons.Batch;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.gpu.GpuContext;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.gpu.silicon.SiliconContext;
import org.brain4j.math.gpu.silicon.SiliconDevice;
import org.brain4j.math.tensor.Tensor;

import java.util.List;

public record Trainer(Model model, TrainingConfig config, List<Monitor> monitors) {

    public static Trainer of(Model model, TrainingConfig config) {
        return new Trainer(model, config, List.of());
    }

    public Trainer {
        if (model == null) throw new IllegalArgumentException("Model cannot be null!");
        if (config == null) throw new IllegalArgumentException("Config cannot be null!");
        
        config.optimizer().initialize();
        config.updater().initialize();
    }
    
    public void fit(ListDataSource dataSource, int epochs) {
        for (int i = 0; i < epochs; i++) {
            fitEpoch(dataSource, i, epochs);
        }
        
        monitors.forEach(x -> x.onEvent(new TrainingEnd()));
    }

    private void fit(ListDataSource dataSource) {
        fitEpoch(dataSource, 0, 1);
        monitors.forEach(x -> x.onEvent(new TrainingEnd()));
    }
    
    private void fitEpoch(ListDataSource dataSource, int index, int total) {
        EpochStart start = new EpochStart(this, index, total);
        monitors.forEach(x -> x.onEvent(start));
        
        dataSource.reset();

        while (dataSource.hasNext()) {
            fitBatch(dataSource);
        }

        Optimizer optimizer = config.optimizer();
        Updater updater = config.updater();

        updater.postFit(optimizer.getLearningRate(), dataSource.getSize());
        
        EpochEnd end = new EpochEnd(this, index, total);
        monitors.forEach(x -> x.onEvent(end));
    }

    public void fitBatch(ListDataSource dataSource) {
        int current = dataSource.getCursor();
        int total = dataSource.getBatches();
        Batch batch = dataSource.nextBatch();

        Tensor[] inputs = batch.getFirst();
        Tensor[] targets = batch.getSecond();
        
        BatchStart start = new BatchStart(this, current, total);
        monitors.forEach(x -> x.onEvent(start));
        
        SiliconDevice device = model.getDevice();
        StatesCache cache = new StatesCache(true);
        
        if (device != null) {
            device.createQueue();
        }

        Tensor[] outputs = model.predict(cache, batch.getFirst());

        Optimizer optimizer = config.optimizer();
        LossFunction loss = config.loss();
        Updater updater = config.updater();

        List<Layer> layers = model.getLayers();

        layers.getLast().computeLoss(cache, targets, outputs, loss);
        layers.forEach(layer -> layer.backward(cache, updater, optimizer));

        int elements = 0;

        for (Tensor input : inputs) {
            elements += input.shapeAt(0);
        }

        optimizer.postBatch();
        updater.postBatch(optimizer.getLearningRate(), elements);

        layers.forEach(Layer::resetGrad);
        
        if (device != null) {
            SiliconContext.finishAndRelease(device.getQueue());
        }
        
        BatchEnd end = new BatchEnd(this, current, total);
        monitors.forEach(x -> x.onEvent(end));
    }
}
