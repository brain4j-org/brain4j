package org.brain4j.core.codec;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

/**
 * JSON specialization of {@link Codec}.
 * Used for {@code BrainFormat} (config.json) and layer/activation/clipper serialization.
 */
public interface JsonCodec<T> extends Codec<T> {
    void write(T value, ObjectNode out);
    T parse(JsonNode in);
}
