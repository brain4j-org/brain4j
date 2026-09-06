package org.brain4j.core.codec.layer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.JsonCodec;
import org.brain4j.core.layer.impl.ReshapeLayer;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.tensor.Shape;

public class ReshapeCodec implements JsonCodec<ReshapeLayer> {
    
    @Override
    public String type() {
        return "reshape";
    }
    
    @Override
    public Class<ReshapeLayer> targetClass() {
        return ReshapeLayer.class;
    }
    
    @Override
    public void write(ReshapeLayer reshapeLayer, ObjectNode out) {
        ArrayNode shape = out.putArray("shape");
        for (int dim : reshapeLayer.config().shape().dims()) {
            shape.add(dim);
        }
    }
    
    @Override
    public ReshapeLayer parse(JsonNode in) {
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

        return new ReshapeLayer(Shape.of(dims));
    }
}
