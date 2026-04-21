package org.brain4j.core.codec.layer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.Codec;
import org.brain4j.core.layer.impl.InputLayer;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.tensor.Shape;

public class InputCodec implements Codec<InputLayer> {
    
    @Override
    public String type() {
        return "input";
    }
    
    @Override
    public Class<InputLayer> targetClass() {
        return InputLayer.class;
    }
    
    @Override
    public void write(InputLayer inputLayer, ObjectNode out) {
        ArrayNode shape = out.putArray("shape");
        for (int dim : inputLayer.shape().dims()) {
            shape.add(dim);
        }
    }
    
    @Override
    public InputLayer parse(JsonNode in) {
        JsonNode shape = in.get("shape");

        if (shape == null || !shape.isArray()) {
            throw Commons.illegalArgument("Shape must be an array");
        }

        int[] dims = new int[shape.size()];

        for (int i = 0; i < shape.size(); i++) {
            JsonNode dim = shape.get(i);

            if (!dim.isInt()) {
                throw Commons.illegalArgument("Shape dimensions must be integers");
            }

            dims[i] = dim.intValue();
        }

        return new InputLayer(Shape.of(dims));
    }
}
