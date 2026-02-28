package org.brain4j.core.codec;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

public interface Codec<T> {
    String type();
    Class<T> targetClass();
    void write(T t, ObjectNode out);
    T parse(JsonNode in);
}
