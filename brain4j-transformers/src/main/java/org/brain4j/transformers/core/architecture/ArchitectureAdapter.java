package org.brain4j.transformers.core.architecture;

import com.google.gson.JsonObject;
import org.brain4j.core.model.Model;
import org.brain4j.math.tensor.Tensor;

import java.util.Map;

public interface ArchitectureAdapter {
    boolean supports(String modelType);
    Model buildModel(JsonObject config, Map<String, Tensor> weights);
}
