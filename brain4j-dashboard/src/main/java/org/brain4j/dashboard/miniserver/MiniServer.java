package org.brain4j.dashboard.miniserver;

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.OutputStream;
import java.lang.reflect.Method;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;

public abstract class MiniServer {
    
    private static final int CORES = Runtime.getRuntime().availableProcessors();
    private static final Executor EXECUTOR = Executors.newWorkStealingPool(CORES);
    private final Map<String, RouteInfo> routes = new HashMap<>();

    public abstract List<Object> getServices();
    
    public void launch(int port) {
        try {
            registerEndpoints();
            HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);
            
            server.setExecutor(EXECUTOR);
            server.createContext("/", exchange -> {
                try {
                    String path = exchange.getRequestURI().getPath();

                    if (!routes.containsKey(path)) {
                        exchange.sendResponseHeaders(404, -1);
                        return;
                    }

                    RouteInfo routeInfo = routes.get(path);
                    Method method = routeInfo.method();
                    Route route = routeInfo.route();

                    Request request = new Request(exchange);
                    boolean isAccepted = false;

                    for (HttpMethod accepted : route.accepted()) {
                        if (accepted == request.method()) {
                            isAccepted = true;
                            break;
                        }
                    }

                    Response response = isAccepted
                        ? (Response) method.invoke(routeInfo.service(), request)
                        : Response.text(400, "Method not supported for this endpoint");

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
        } catch (IOException e) {
            e.printStackTrace(System.err);
            throw new RuntimeException(e);
        }
    }
    
    protected Response serveResource(String resourceName, String contentType) {
        try (var stream = getClass().getClassLoader().getResourceAsStream(resourceName)) {
            if (stream == null) {
                return Response.json(404, Map.of("message", "Resource not found"));
            }
            
            String resource = new String(stream.readAllBytes(), StandardCharsets.UTF_8);
            return Response.text(200, resource).addHeader("Content-Type", contentType);
        } catch (IOException e) {
            return Response.json(500, Map.of("message", "Cannot load resource"));
        }
    }

    private void registerEndpoints() {
        for (Object service : getServices()) {
            for (Method method : service.getClass().getDeclaredMethods()) {
                if (method.isAnnotationPresent(Route.class)) {
                    Route route = method.getAnnotation(Route.class);
                    String path = route.value();
                    Class<?> returnType = method.getReturnType();

                    if (returnType != Response.class) {
                        throw new IllegalArgumentException("Endpoint %s has is missing a response!");
                    }

                    routes.put(path, new RouteInfo(service, method, route));
                }
            }
        }
    }

    private record RouteInfo(Object service, Method method, Route route) {
    }
}
