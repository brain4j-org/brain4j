package org.brain4j.dashboard.server;

import org.brain4j.dashboard.miniserver.MiniServer;
import org.brain4j.dashboard.miniserver.Request;
import org.brain4j.dashboard.miniserver.Response;
import org.brain4j.dashboard.miniserver.Route;

import java.util.Map;

public class BrainServer extends MiniServer {
    
    public BrainServer(int port) {
        super(port);
    }
    
    @Override
    public Object getService() {
        return this;
    }
    
    @Route("/hello")
    public Response hello(Request request) {
        return Response.json(200, Map.of("method", request.method(), "params", request.queryParams()));
    }
}
