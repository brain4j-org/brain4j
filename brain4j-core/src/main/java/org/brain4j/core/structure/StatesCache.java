package org.brain4j.core.structure;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.transformers.head.AttentionHead;
import org.brain4j.math.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class StatesCache {

    private final Map<Integer, List<Tensor>> feedForwardCache;
    private final Map<Integer, List<Tensor>> keyCache;
    private final Map<Integer, List<Tensor>> valueCache;

    private final Tensor[] weightsGradientCache;
    private final Tensor[] biasGradientCache;
    private final Tensor[] inputTensorsCache;
    private final Tensor[] outputTensorsCache;

    public StatesCache() {
        this.weightsGradientCache = new Tensor[Layer.getTotalLayers()];
        this.biasGradientCache = new Tensor[Layer.getTotalLayers()];
        this.inputTensorsCache = new Tensor[Layer.getTotalLayers()];
        this.outputTensorsCache = new Tensor[Layer.getTotalLayers()];
        this.feedForwardCache = new ConcurrentHashMap<>(); // TODO: Migrate to arrays
        this.keyCache = new ConcurrentHashMap<>();
        this.valueCache = new ConcurrentHashMap<>();

        markAsNewSession();
    }

    public void gradientChange(Layer layer, Tensor weightChange, Tensor biasChange) {
        weightsGradientCache[layer.getId()] = weightChange;
        biasGradientCache[layer.getId()] = biasChange;
    }

    public void setInputTensor(Layer layer, Tensor value) {
        inputTensorsCache[layer.getId()] = value;
    }

    public Tensor getInputTensor(Layer layer) {
        return inputTensorsCache[layer.getId()];
    }

    public void setOutputTensor(Layer layer, Tensor value) {
        outputTensorsCache[layer.getId()] = value;
    }

    public Tensor getOutputTensor(Layer layer) {
        return outputTensorsCache[layer.getId()];
    }
    
    public void markAsNewSession() {
        keyCache.clear();
        valueCache.clear();
    }

    // avoid shape mismatch errors
    public boolean isCompatibleWithCache(Tensor tensor) {
        if (keyCache.isEmpty() && valueCache.isEmpty()) {
            return true;
        }

        for (List<Tensor> tensors : keyCache.values()) {
            if (tensors.isEmpty()) continue;

            Tensor cached = tensors.getFirst();
            if (cached.shape()[0] != tensor.shape()[0]) {
                return false;
            }
        }

        
        return true;
    }

    public List<Tensor> getFeedForwardForLayer(Layer layer) {
        return feedForwardCache.computeIfAbsent(layer.hashCode(), k -> new ArrayList<>());
    }
    
    public List<Tensor> getKeyCacheForHead(AttentionHead head) {
        return keyCache.computeIfAbsent(head.hashCode(), k -> new ArrayList<>());
    }
    
    public List<Tensor> getValueCacheForHead(AttentionHead head) {
        return valueCache.computeIfAbsent(head.hashCode(), k -> new ArrayList<>());
    }
}
