package org.brain4j.dashboard.miniserver;

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Request {
    
    private final HttpExchange exchange;

    public Request(HttpExchange exchange) {
        this.exchange = exchange;
    }
    
    public Headers headers() {
        return exchange.getRequestHeaders();
    }
    
    public HttpMethod method() {
        return HttpMethod.valueOf(exchange.getRequestMethod());
    }

    public String path() {
        return exchange.getRequestURI().getPath();
    }
    
    
    public Map<String, String> queryParams() {
        return parseQuery(exchange.getRequestURI().getQuery());
    }
    
    public String body() {
        try {
            return new String(exchange.getRequestBody().readAllBytes());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
    
    private Map<String, String> parseQuery(String query) {
        Map<String, String> map = new HashMap<>();
        
        if (query == null) return map;
        
        for (String pair : query.split("&")) {
            String[] parts = pair.split("=");
            if (parts.length == 2) {
                map.put(parts[0], parts[1]);
            }
        }
        
        return map;
    }
}
