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
import org.brain4j.math.gpu.silicon.SiliconContext;
import org.brain4j.math.gpu.silicon.SiliconDevice;
import org.brain4j.math.tensor.Tensor;

import java.util.List;

public final class DefaultTrainer implements Trainer {
    
    private final Model model;
    private final TrainingConfig config;
    private final List<Monitor> monitors;
    
    DefaultTrainer(Model model, TrainingConfig config, List<Monitor> monitors) {
        if (model == null) throw new IllegalArgumentException("Model cannot be null!");
        if (config == null) throw new IllegalArgumentException("Config cannot be null!");
        
        config.optimizer().initialize();
        config.updater().initialize();
        
        this.model = model;
        this.config = config;
        this.monitors = monitors;
    }
    
    public void fit(ListDataSource dataSource, int epochs) {
        for (int i = 0; i < epochs; i++) {
            fitEpoch(dataSource, i, epochs);
        }
        
        monitors.forEach(x -> x.onEvent(new TrainingEnd()));
    }
    
    public void fit(ListDataSource dataSource) {
        fitEpoch(dataSource, 0, 1);
        monitors.forEach(x -> x.onEvent(new TrainingEnd()));
    }
    
    private void fitEpoch(ListDataSource dataSource, int index, int total) {
        EpochStart epochStart = new EpochStart(this, index, total);
        monitors.forEach(x -> x.onEvent(epochStart));
        
        dataSource.reset();
        int totalBatches = dataSource.getBatches();
        
        while (dataSource.hasNext()) {
            int cursor = dataSource.getCursor();
            Batch batch = dataSource.nextBatch();
            
            BatchStart batchStart = new BatchStart(this, cursor, totalBatches);
            monitors.forEach(x -> x.onEvent(batchStart));
            
            fitBatch(batch, cursor, totalBatches);
            
            BatchEnd end = new BatchEnd(this, cursor, totalBatches);
            monitors.forEach(x -> x.onEvent(end));
        }
        
        Optimizer optimizer = config.optimizer();
        Updater updater = config.updater();
        
        updater.postFit(optimizer.getLearningRate(), dataSource.getSize());
        
        EpochEnd end = new EpochEnd(this, index, total);
        monitors.forEach(x -> x.onEvent(end));
    }
    
    public void fitBatch(Batch batch, int cursor, int totalBatches) {
        Tensor[] inputs = batch.getFirst();
        
        SiliconDevice device = model.getDevice();
        StatesCache cache = new StatesCache(true);
        
        if (device != null) {
            device.createQueue();
        }
        
        Tensor[] outputs = forward(cache, inputs);
        backward(cache, batch, outputs);
        resetGrad();
        
        if (device != null) {
            SiliconContext.finishAndRelease(device.getQueue());
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
        
        layers.getLast().computeLoss(cache, targets, outputs, loss);
        layers.forEach(layer -> layer.backward(cache, updater, optimizer));
        
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
        return monitors;
    }
}
