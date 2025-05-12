package org.brain4j.core.structure;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.transformers.head.AttentionHead;
import org.brain4j.math.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class StatesCache {

    private final Map<Integer, Tensor> feedForwardCache;
    private final Map<Integer, List<Tensor>> keyCache;
    private final Map<Integer, List<Tensor>> valueCache;

    private final Tensor[] inputTensorsCache;
    private final Tensor[] outputTensorsCache;

    public StatesCache() {
        this.inputTensorsCache = new Tensor[Layer.getTotalLayers()];
        this.outputTensorsCache = new Tensor[Layer.getTotalLayers()];
        this.feedForwardCache = new ConcurrentHashMap<>(); // TODO: Migrate to arrays
        this.keyCache = new ConcurrentHashMap<>();
        this.valueCache = new ConcurrentHashMap<>();

        markAsNewSession();
    }

    public void setInputTensor(Layer layer, Tensor value) {
        inputTensorsCache[layer.getId()] = value;
    }

    public Tensor getInputTensor(int index) {
        return inputTensorsCache[index];
    }

    public void setOutputTensor(Layer layer, Tensor value) {
        outputTensorsCache[layer.getId()] = value;
    }

    public Tensor getOutputTensor(int index) {
        return outputTensorsCache[index];
    }
    
    public void markAsNewSession() {
        keyCache.clear();
        valueCache.clear();
    }

    // Avoid shape mismatch errors
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

    public Tensor getFeedForwardCache(int offset, Layer layer) {
        return feedForwardCache.computeIfAbsent(layer.hashCode() + offset, k -> null);
    }

    public void setFeedForwardCache(
        int offset,
        Layer layer,
        Tensor tensor
    ) {
        feedForwardCache.put(layer.hashCode() + offset, tensor);
    }
    
    public List<Tensor> getKeyCacheForHead(AttentionHead head) {
        return keyCache.computeIfAbsent(head.hashCode(), k -> new ArrayList<>());
    }
    
    public List<Tensor> getValueCacheForHead(AttentionHead head) {
        return valueCache.computeIfAbsent(head.hashCode(), k -> new ArrayList<>());
    }
}
