package org.brain4j.dashboard.miniserver;

import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

public class Response {
    
    private final Map<String, String> headers = new HashMap<>();
    private int status;
    private byte[] body;
    
    public static Response text(int status, String body) {
        Response r = new Response();
        r.status = status;
        r.body = body.getBytes(StandardCharsets.UTF_8);
        r.headers.put("Content-Type", "text/plain; charset=utf-8");
        return r;
    }
    
    public static Response json(int status, Object obj) {
        Response r = new Response();
        r.status = status;
        r.body = Json.serialize(obj);
        r.headers.put("Content-Type", "application/json");
        return r;
    }

    public static Response html(int status, String body) {
        Response r = new Response();
        r.status = status;
        r.body = body.getBytes(StandardCharsets.UTF_8);
        r.headers.put("Content-Type", "text/html; charset=utf-8");
        return r;
    }

    public static Response ok() {
        return text(200, "");
    }

    public Response addHeader(String key, String value) {
        headers.put(key, value);
        return this;
    }
    
    public int status() {
        return status;
    }
    
    public byte[] body() {
        return body;
    }
    
    public Map<String, String> headers() {
        return headers;
    }
}
