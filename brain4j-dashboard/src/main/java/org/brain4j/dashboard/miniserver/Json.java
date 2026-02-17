package org.brain4j.dashboard.miniserver;

import com.fasterxml.jackson.databind.ObjectMapper;

public class Json {
    
    private static final ObjectMapper MAPPER = new ObjectMapper();

    public static byte[] serialize(Object obj) {
        try {
            return MAPPER.writeValueAsBytes(obj);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
