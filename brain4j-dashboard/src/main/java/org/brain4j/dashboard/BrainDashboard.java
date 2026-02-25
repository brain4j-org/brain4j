package org.brain4j.dashboard;

import org.brain4j.core.model.Model;
import org.brain4j.core.model.impl.Sequential;
import org.brain4j.core.monitor.Monitor;
import org.brain4j.core.training.Trainer;
import org.brain4j.dashboard.server.BrainServer;
import org.brain4j.math.data.ListDataSource;

import java.util.function.Supplier;

public record BrainDashboard(Model model, Trainer trainer) {
    
    public <T extends Monitor> T attach(Class<T> clazz, Supplier<T> constructor) {
        T monitor = trainer.getMonitor(clazz);
        
        if (monitor == null) {
            monitor = constructor.get();
            trainer.attach(monitor);
        }
        
        return monitor;
    }
    
    public void launch(ListDataSource trainSet, ListDataSource testSet, int epochs, int port) {
        BrainServer server = new BrainServer(this, trainSet, testSet, epochs);
        server.launch(port);
    }
}
