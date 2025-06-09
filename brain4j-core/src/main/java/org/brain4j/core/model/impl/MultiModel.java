package org.brain4j.core.model.impl;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.model.Model;
import org.brain4j.core.training.StatesCache;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.core.training.wrappers.EvaluationResult;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.device.DeviceType;
import org.brain4j.math.tensor.Tensor;

import java.util.Iterator;
import java.util.List;

public class MultiModel implements Model {

    protected List<Model> models;
    protected List<Layer> layers;

    public static Model of(List<Model> models, Layer... layers) {
        return new MultiModel(models, layers);
    }

    protected MultiModel(List<Model> models, Layer... layers) {
        this.models = models;
        this.layers = List.of(layers);
    }

    @Override
    public Model add(Layer layer) {
        return null;
    }

    @Override
    public Model add(int index, Layer layer) {
        return null;
    }

    @Override
    public Tensor predict(Tensor input) {
        return null;
    }

    @Override
    public Tensor predict(StatesCache cache, Tensor input) {
        return null;
    }

    @Override
    public Tensor predict(StatesCache cache, Tensor input, boolean training) {
        return null;
    }

    @Override
    public void fit(ListDataSource train, ListDataSource validation, int epoches, int evaluateEvery) {

    }

    @Override
    public EvaluationResult evaluate(ListDataSource dataSource) {
        return null;
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
    public Model to(DeviceType deviceType) {
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
    public Optimizer optimizer() {
        return null;
    }

    @Override
    public Updater updater() {
        return null;
    }

    @Override
    public LossFunction lossFunction() {
        return null;
    }

    @Override
    public void summary() {

    }

    @Override
    public void zeroGrad() {

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
