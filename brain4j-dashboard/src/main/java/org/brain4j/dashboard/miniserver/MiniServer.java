package org.brain4j.dashboard.miniserver;

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.OutputStream;
import java.lang.reflect.Method;
import java.net.InetSocketAddress;
import java.util.HashMap;
import java.util.Map;

public abstract class MiniServer {
    
    private final Map<String, Method> routes = new HashMap<>();
    private final HttpServer server;
    
    public MiniServer(int port) {
        try {
            this.server = HttpServer.create(new InetSocketAddress(port), 0);
            registerEndpoints();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
    
    public abstract Object getService();
    
    public void launch() {
        server.createContext("/", exchange -> {
            try {
                String path = exchange.getRequestURI().getPath();
                
                if (!routes.containsKey(path)) {
                    exchange.sendResponseHeaders(404, -1);
                    return;
                }
                
                Method method = routes.get(path);
                Response response = (Response) method.invoke(
                    getService(), new Request(exchange)
                );
                
                byte[] respBody = response.body();
                Headers respHeaders = exchange.getResponseHeaders();
                
                response.headers().forEach(respHeaders::add);
                exchange.sendResponseHeaders(
                    response.status(),
                    respBody.length
                );
                
                try (OutputStream os = exchange.getResponseBody()) {
                    os.write(response.body());
                }
            } catch (Exception e) {
                e.printStackTrace(System.err);
                exchange.sendResponseHeaders(500, -1);
            }
        });
        
        server.start();
    }
    
    private void registerEndpoints() {
        for (Method method : getService().getClass().getDeclaredMethods()) {
            if (method.isAnnotationPresent(Route.class)) {
                String path = method.getAnnotation(Route.class).value();
                Class<?> returnType = method.getReturnType();
                
                if (returnType != Response.class) {
                    throw new IllegalArgumentException("Endpoint %s has is missing a response!");
                }
                
                routes.put(path, method);
            }
        }
    }
}
