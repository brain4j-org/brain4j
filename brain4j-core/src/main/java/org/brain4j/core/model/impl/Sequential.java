package org.brain4j.core.model.impl;

import org.brain4j.core.Brain4J;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.Node;
import org.brain4j.core.layer.impl.InputLayer;
import org.brain4j.core.model.Model;
import org.brain4j.core.model.ModelBlock;
import org.brain4j.core.model.ModelSpecs;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.gpu.silicon.SiliconDevice;
import org.brain4j.math.tensor.Tensor;

import java.text.DecimalFormat;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

public class Sequential implements Model, ModelBlock {
    
    private final Graph graph;
    private final ModelSpecs specs;
    private final List<Layer> layers;
    private final SiliconDevice device;
    
    public Sequential(ModelSpecs specs, SiliconDevice device, List<Layer> layers, int seed) {
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
        
        this.graph = Graph.of(seed, lastNode);
    }

    public Sequential(ModelSpecs specs, SiliconDevice device, int seed) {
        this(specs, device, specs.buildLayerList(), seed);
    }
    
    @Override
    public Tensor[] predict(StatesCache cache, Tensor... inputs) {
        if (device != null && !cache.isTraining()) {
            device.createResources();
        }
        
        Tensor[] out = graph.predict(cache, inputs);
        
        if (device != null && !cache.isTraining()) {
            device.closeResources();
        }
        
        return out;
    }

    @Override
    public Sequential fork(SiliconDevice device) {
        List<Layer> copiedLayers = layers.stream().map(Layer::copy).toList();
        copiedLayers.forEach(x -> x.to(device));
        return new Sequential(specs.copy(), device, copiedLayers, graph.seed());
    }
    
    @Override
    public SiliconDevice device() {
        return device;
    }

    @Override
    public void summary() {
        Brain4J.fixConsole();
        
        StringBuilder stats = new StringBuilder();
        DecimalFormat format = new DecimalFormat("#,###");
        
        String pattern = "%-7s %-20s %-15s %-13s %-15s\n";
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
        
        String formattedTotal = format.format(totalParameters);
        String formattedTrainable = format.format(trainableParameters);
        
        stats.append(Commons.getHeader(" Recap ", Commons.HEADER_CHAR));
        stats.append("Total parameters: %s (%s)\n".formatted(formattedTotal, sizeOfTotalParams));
        stats.append("Trainable parameters: %s (%s)\n".formatted(formattedTrainable, sizeOfTrainableParams));
        stats.append(Commons.getHeader("", Commons.HEADER_CHAR));
        
        Arrays.stream(stats.toString().split("\n")).forEach(System.out::println);
    }
    
    @Override
    public Sequential copy() {
        List<Layer> copiedLayers = layers.stream()
            .map(Layer::copy)
            .toList();
        
        return new Sequential(specs.copy(), device, copiedLayers, graph.seed());
    }
    
    @Override
    public void appendTo(List<Layer> layers) {
        layers.addAll(getLayers());
    }
    
    @Override
    public List<Layer> getLayers() {
        return Collections.unmodifiableList(layers);
    }
    
    public ModelSpecs specs() {
        return specs;
    }

    public Graph graph() {
        return graph;
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
            int trainable = layer.calculateTrainableParams();

            Tensor weightsTensor = layer.getParam("weights");
            
            String shape = weightsTensor == null ? "N/A" : Arrays.toString(weightsTensor.shape());
            String row = pattern.formatted(i, layerType, shape, format.format(total), layer.activation().name());
            
            builder.append(row);
            
            totalParams.addAndGet(total);
            trainableParams.addAndGet(trainable);
        }
    }
}
