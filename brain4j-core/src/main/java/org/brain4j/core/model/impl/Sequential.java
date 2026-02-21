package org.brain4j.core.model.impl;

import org.brain4j.core.Brain4J;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.Node;
import org.brain4j.core.layer.newimpl.InputLayer;
import org.brain4j.math.loss.LossFunction;
import org.brain4j.math.loss.impl.BinaryCrossEntropy;
import org.brain4j.core.model.Model;
import org.brain4j.core.model.ModelBlock;
import org.brain4j.core.model.ModelSpecs;
import org.brain4j.core.training.wrappers.EvaluationResult;
import org.brain4j.math.Tensors;
import org.brain4j.math.commons.Batch;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.gpu.silicon.SiliconDevice;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.commons.Range;

import java.text.DecimalFormat;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

public class Sequential implements Model, ModelBlock {
    
    private final DAG dag;
    private final ModelSpecs specs;
    private final List<Layer> layers;
    private final SiliconDevice device;
    
    private Sequential(ModelSpecs specs, SiliconDevice device, List<Layer> layers, int seed) {
        this.specs = specs;
        this.device = device;
        this.layers = layers;
        
        if (layers.isEmpty()) {
            throw Commons.illegalArgument("Layer list is empty!");
        }
        
        Layer first = layers.getFirst();
        
        if (!(first instanceof InputLayer inputLayer)) {
            throw Commons.illegalArgument("First layer is not an input layer!");
        }
        
        Node lastNode = Node.input(inputLayer.shape());
        
        for (int i = 1; i < layers.size(); i++) {
            Layer current = layers.get(i);
            lastNode = current.apply(lastNode);
        }
        
        this.dag = DAG.of(seed, lastNode);
    }
    
    public Sequential(ModelSpecs specs, SiliconDevice device, int seed) {
        this(specs, device, specs.buildLayerList(), seed);
    }
    
    @Override
    public Tensor[] predict(StatesCache cache, Tensor... inputs) {
        if (device != null && !cache.isTraining()) {
            device.createResources();
        }
        
        Tensor[] out = dag.predict(cache, inputs);
        
        if (device != null && !cache.isTraining()) {
            device.closeResources();
        }
        
        return out;
    }
    
    @Override
    public EvaluationResult evaluate(ListDataSource dataSource, LossFunction lossFunction) {
        int classes = Math.max(2, dataSource.getSamples().getFirst().getLabel(0).elements());
        Map<Integer, Tensor> classifications = new HashMap<>();

        if (!lossFunction.isRegression()) {
            for (int i = 0; i < classes; i++) {
                classifications.put(i, Tensors.zeros(classes));
            }
        }
        
        AtomicReference<Double> totalLoss = new AtomicReference<>(0.0);
        
        dataSource.reset();
        
        while (dataSource.hasNext()) {
            Batch batch = dataSource.nextBatch();
            makeEvaluation(batch, classifications, totalLoss, lossFunction);
        }
        
        return new EvaluationResult(totalLoss.get() / dataSource.getSize(), classes, classifications);
    }
    
    @Override
    public double loss(ListDataSource dataSource, LossFunction lossFunction) {
        Map<Integer, Tensor> classifications = new HashMap<>();
        AtomicReference<Double> totalLoss = new AtomicReference<>(0.0);
        
        dataSource.reset();
        
        while (dataSource.hasNext()) {
            Batch batch = dataSource.nextBatch();
            makeEvaluation(batch, classifications, totalLoss, lossFunction);
        }
        
        return totalLoss.get() / dataSource.getSize();
    }
    
    @Override
    public Sequential fork(SiliconDevice device) {
        return null; // TODO
//        List<Layer0> copiedLayers = layers.stream().map(Layer::clone).toList();
//        copiedLayers.forEach(x -> x.toDevice(device));
//        return new Sequential(specs.clone(), device, new ArrayList<>(copiedLayers), seed);
    }
    
    @Override
    public SiliconDevice getDevice() {
        return device;
    }
    
    @Override
    public void summary() {
        Brain4J.fixConsole();
        
        StringBuilder stats = new StringBuilder();
        DecimalFormat format = new DecimalFormat("#,###");
        
        String pattern = "%-7s %-20s %-12s %-15s %-15s\n";
        String divider = Commons.getHeader(" Architecture ", Commons.HEADER_CHAR);
        
        stats.append(divider);
        stats.append(pattern.formatted("Index", "Layer Type", "Weights Shape", "Parameters", "Activation")).append("\n");
        
        AtomicLong totalParams = new AtomicLong(0);
        AtomicLong trainableParams = new AtomicLong(0);
        
        append(pattern, stats, format, totalParams, trainableParams);
        
        byte floatSize = Float.BYTES; // 4 bytes
        
        long totalParameters = totalParams.get();
        long trainableParameters = trainableParams.get();
        
        String sizeOfTotalParams = Commons.formatNumber(totalParams.get() * floatSize);
        String sizeOfTrainableParams = Commons.formatNumber(trainableParams.get() * floatSize);
        
        stats.append(Commons.getHeader(" Recap ", Commons.HEADER_CHAR));
        stats.append("Total parameters: %s (%s)\n".formatted(totalParameters, sizeOfTotalParams));
        stats.append("Trainable parameters: %s (%s)\n".formatted(trainableParameters, sizeOfTrainableParams));
        stats.append(Commons.getHeader("", Commons.HEADER_CHAR));
        
        Arrays.stream(stats.toString().split("\n")).forEach(System.out::println);
    }
    
    public ModelSpecs getSpecs() {
        return specs;
    }
    
    public List<Layer> getLayers() {
        return Collections.unmodifiableList(layers);
    }
    
    private void makeEvaluation(
        Batch batch,
        Map<Integer, Tensor> classifications,
        AtomicReference<Double> totalLoss,
        LossFunction lossFunction
    ) {
        Tensor[] inputs = batch.getFirst();
        Tensor[] labels = batch.getSecond();
        
        if (device != null) device.createResources();
        
        StatesCache cache = new StatesCache(false);
        Tensor[] outputs = predict(cache, inputs);
        
        for (Tensor input : inputs) {
            int batchSize = input.shapeAt(0);
            
            for (int i = 0; i < outputs.length; i++) {
                Tensor output = outputs[i].to(null); // GPU -> CPU
                Tensor label = labels[i].to(null);   // GPU -> CPU
                
                for (int b = 0; b < batchSize; b++) {
                    Range range = Range.point(b);
                    
                    Tensor sampleOutput = output.slice(range).flatten();
                    Tensor sampleLabel = label.slice(range).flatten();
                    
                    int predIndex = sampleOutput.argmax();
                    int targetIndex = sampleLabel.argmax();
                    
                    if (sampleOutput.elements() == 1 && lossFunction instanceof BinaryCrossEntropy) {
                        predIndex = sampleOutput.get(0) > 0.5 ? 1 : 0;
                        targetIndex = (int) sampleLabel.get(0);
                    }
                    
                    double loss = lossFunction.calculate(sampleLabel, sampleOutput);
                    totalLoss.updateAndGet(v -> v + loss);
                    
                    Tensor predictions = classifications.get(targetIndex);
                    
                    if (predictions != null) {
                        int pred = (int) predictions.get(predIndex);
                        predictions.set(pred + 1, predIndex);
                    }
                }
            }
        }
        
        if (device != null) device.closeResources();
    }
    
    private void append(
        String pattern,
        StringBuilder builder,
        DecimalFormat format,
        AtomicLong totalParams,
        AtomicLong trainableParams
    ) {
        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            String layerType = layer.getClass().getSimpleName();
            
            int total = layer.calculateTotalParameters();
            Tensor weightsTensor = layer.getParam("weights");
            
            String shape = weightsTensor == null ? "N/A" : Arrays.toString(weightsTensor.shape());
            String row = pattern.formatted(i, layerType, shape, format.format(total), layer.activation().name());
            
            builder.append(row);
            
            totalParams.addAndGet(total);
            if (!layer.frozen()) trainableParams.addAndGet(total);
        }
    }
    
    @Override
    public Sequential copy() {
        List<Layer> copiedLayers = layers.stream()
            .map(Layer::copy)
            .toList();
        
        return new Sequential(specs.copy(), device, copiedLayers, dag.seed());
    }
    
    @Override
    public void appendTo(List<Layer> layers) {
        layers.addAll(getLayers());
    }
}
