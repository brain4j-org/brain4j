package org.brain4j.core.codec.layer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.codec.Codec;
import org.brain4j.core.layer.impl.utility.ActivationLayer;
import org.brain4j.math.activation.impl.Linear;

public class ActivationCodec implements Codec<ActivationLayer> {

    @Override
    public String type() {
        return "activation";
    }

    @Override
    public Class<ActivationLayer> targetClass() {
        return ActivationLayer.class;
    }

    @Override
    public void write(ActivationLayer activationLayer, ObjectNode out) {
    }

    @Override
    public ActivationLayer parse(JsonNode in) {
        return new ActivationLayer(new Linear());
    }
}
