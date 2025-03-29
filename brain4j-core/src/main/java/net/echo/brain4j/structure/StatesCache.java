package net.echo.brain4j.structure;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.transformers.head.AttentionHead;
import net.echo.math4j.math.tensor.Tensor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class StatesCache {

    private final Tensor[] inputTensorsCache;
    private final Tensor[] outputTensorsCache;
    private boolean isNewSession = true;
    
    private final Map<Integer, List<Tensor>> keyCache;
    private final Map<Integer, List<Tensor>> valueCache;

    public StatesCache() {
        this.inputTensorsCache = new Tensor[Parameters.TOTAL_LAYERS];
        this.outputTensorsCache = new Tensor[Parameters.TOTAL_LAYERS];
        this.keyCache = new HashMap<>();
        this.valueCache = new HashMap<>();
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
    
    public boolean isNewSession() {
        if (isNewSession) {
            isNewSession = false;
            return true;
        }
        return false;
    }
    
    public void markAsNewSession() {
        isNewSession = true;
        clearKVCache();
    }
    
    public List<Tensor> getKeyCacheForHead(AttentionHead head) {
        return keyCache.computeIfAbsent(head.hashCode(), k -> new ArrayList<>());
    }
    
    public List<Tensor> getValueCacheForHead(AttentionHead head) {
        return valueCache.computeIfAbsent(head.hashCode(), k -> new ArrayList<>());
    }
    
    public void clearKVCache() {
        keyCache.clear();
        valueCache.clear();
    }
}
