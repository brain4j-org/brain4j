package net.echo.brain4j.structure;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.transformers.head.AttentionHead;
import net.echo.math4j.math.tensor.Tensor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class StatesCache {

    private final Map<Integer, List<Tensor>> feedForwardCache;
    private final Map<Integer, List<Tensor>> keyCache;
    private final Map<Integer, List<Tensor>> valueCache;

    private final Tensor[] inputTensorsCache;
    private final Tensor[] outputTensorsCache;

    public StatesCache() {
        this.inputTensorsCache = new Tensor[Parameters.TOTAL_LAYERS];
        this.outputTensorsCache = new Tensor[Parameters.TOTAL_LAYERS];
        this.feedForwardCache = new HashMap<>();
        this.keyCache = new HashMap<>();
        this.valueCache = new HashMap<>();

        markAsNewSession();
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
            if (!tensors.isEmpty()) {
                Tensor cached = tensors.get(0);
                if (cached.shape()[0] != tensor.shape()[0]) {
                    return false;
                }
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
