package org.brain4j.core.layer.impl;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import org.brain4j.core.layer.OldLayer;
import org.brain4j.math.commons.JsonAdapter;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.scaler.FeatureScaler;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.AutogradContext;

import java.util.HashSet;
import java.util.Set;

import static org.brain4j.core.importing.Registries.SCALER_REGISTRY;

public class ScalingLayer extends OldLayer {

    private FeatureScaler scaler;
    private Set<Integer> enabledInputs;
    private int dimension;

    private ScalingLayer() {
    }
    
    public ScalingLayer(FeatureScaler scaler) {
        this.scaler = scaler;
        this.enabledInputs = null;
    }
    
    public ScalingLayer(FeatureScaler scaler, Set<Integer> enabledInputs) {
        this.scaler = scaler;
        this.enabledInputs = enabledInputs;
    }

    @Override
    public void connect() {
        this.dimension = previous.size();
    }

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        boolean scaleAll = enabledInputs == null;
        
        if (!scaleAll) {
            for (int i : enabledInputs) {
                if (i < 0 || i >= inputs.length) {
                    throw Commons.illegalState("Enabled input index out of range: %s", i);
                }
            }
        }
        
        Tensor[] outputs = new Tensor[inputs.length];
        
        for (int i = 0; i < outputs.length; i++) {
            Tensor input = inputs[i];
            
            if (!scaleAll && !enabledInputs.contains(i)) {
                outputs[i] = input;
                continue;
            }
            
            Tensor result = scaler.transform(input);

            AutogradContext context = input.getAutogradContext();
            result.setAutogradContext(context);

            outputs[i] = result;
        }

        return outputs;
    }

    @Override
    public int size() {
        return dimension;
    }

    @Override
    public void serialize(JsonObject object) {
        JsonArray array = new JsonArray();
        
        for (int x : enabledInputs) array.add(x);
        
        object.addProperty("scaler", SCALER_REGISTRY.fromClass(scaler.getClass()));
        object.add("enabled_inputs", array);

        if (scaler instanceof JsonAdapter adapter) {
            adapter.serialize(object);
        }
    }

    @Override
    public void deserialize(JsonObject object) {
        String scalerType = object.getAsJsonPrimitive("scaler").getAsString();
        JsonArray enabled = object.getAsJsonArray("enabled_inputs");
        
        this.enabledInputs = new HashSet<>();
        this.scaler = SCALER_REGISTRY.toInstance(scalerType);
        
        for (int i = 0; i < enabled.size(); i++) {
            enabledInputs.add(enabled.get(i).getAsInt());
        }
        
        if (scaler instanceof JsonAdapter adapter) {
            adapter.deserialize(object);
        }
    }
}
