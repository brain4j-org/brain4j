package org.brain4j.dashboard;

import org.brain4j.core.model.Model;
import org.brain4j.core.training.Trainer;
import org.brain4j.dashboard.server.BrainServer;

public class BrainDashboard {
    
    private final Model model;
    private final Trainer trainer;
    
    public BrainDashboard(Model model, Trainer trainer) {
        this.model = model;
        this.trainer = trainer;
    }
    
    public void launch(int port) {
        BrainServer server = new BrainServer(port);
        server.launch();
    }
}
