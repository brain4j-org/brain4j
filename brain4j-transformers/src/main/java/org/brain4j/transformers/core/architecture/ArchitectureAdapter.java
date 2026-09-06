package org.brain4j.transformers.core.architecture;

import com.fasterxml.jackson.databind.JsonNode;
import org.brain4j.core.model.Model;
import org.brain4j.math.tensor.Tensor;

import java.util.Map;

public interface ArchitectureAdapter {
    boolean supports(String modelType);
    Model buildModel(JsonNode config, Map<String, Tensor> weights);
}
