package org.brain4j.core.model.impl;

import org.brain4j.common.data.ListDataSource;
import org.brain4j.common.device.Device;
import org.brain4j.common.tensor.Tensor;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.model.Model;
import org.brain4j.core.training.StatesCache;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.core.training.wrappers.EvaluationResult;

import java.util.Iterator;
import java.util.List;
import java.util.function.Consumer;

public class ComputationalGraph implements Model {
    @Override
    public Model add(Layer layer) {
        throw new UnsupportedOperationException("Not supported for this class.");
    }
    
    @Override
    public Model add(int index, Layer layer) {
        throw new UnsupportedOperationException("Not supported for this class.");
    }
    
    @Override
    public Tensor predict(StatesCache cache, boolean training, Tensor... inputs) {
        return null;
    }
    
    @Override
    public void backpropagate(StatesCache cache, Tensor outputs, Tensor targets) {
        throw new UnsupportedOperationException("Not supported for this class.");
    }
    
    @Override
    public void updateWeights(Consumer<Layer> callback) {
        throw new UnsupportedOperationException("Not supported for this class.");
    }
    
    @Override
    public void fit(ListDataSource train, ListDataSource validation, int epoches, int evaluateEvery) {
        throw new UnsupportedOperationException("Not supported for this class.");
    }
    
    @Override
    public EvaluationResult evaluate(ListDataSource dataSource) {
        return null; // TODO
    }
    
    @Override
    public double loss(ListDataSource dataSource) {
        return 0;
    }
    
    @Override
    public Model compile(LossFunction lossFunction, Optimizer optimizer, Updater updater) {
        return null;
    }

    @Override
    public Model to(Device device) {
        return null;
    }

    @Override
    public Device device() {
        return null;
    }
    
    @Override
    public List<Layer> layers() {
        return List.of();
    }
    
    @Override
    public List<Layer> flattened() {
        return List.of();
    }
    
    @Override
    public Layer layerAt(int index) {
        return null;
    }
    
    @Override
    public Layer flattenedAt(int index) {
        return null;
    }
    
    @Override
    public void summary() {
    
    }
    
    @Override
    public void zeroGrad() {
        throw new UnsupportedOperationException("Not supported for this class.");
    }
    
    @Override
    public int size() {
        return 0;
    }
    
    @Override
    public Iterator<Layer> iterator() {
        return null;
    }
}
