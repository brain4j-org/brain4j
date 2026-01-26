package org.brain4j.core.training;

import org.brain4j.core.model.Model;
import org.brain4j.core.monitor.Monitor;
import org.brain4j.math.commons.Batch;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Tensor;

import java.util.List;

public interface Trainer {
    
    static DefaultTrainer of(Model model, TrainingConfig config) {
        return new DefaultTrainer(model, config, List.of());
    }
    
    static DefaultTrainer of(Model model, TrainingConfig config, List<Monitor> monitors) {
        return new DefaultTrainer(model, config, monitors);
    }
    
    void fit(ListDataSource dataSource, int epochs);
    void fit(ListDataSource dataSource);
    void fitBatch(Batch batch, int index, int totalBatches);

    Tensor[] forward(StatesCache cache, Tensor[] inputs);
    void backward(StatesCache cache, Batch batch, Tensor[] outputs);
    void resetGrad();

    TrainingConfig config();
    Model model();
    List<Monitor> monitors();
}
