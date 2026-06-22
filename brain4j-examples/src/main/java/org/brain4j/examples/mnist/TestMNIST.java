package org.brain4j.examples.mnist;

import org.brain4j.core.layer.impl.DenseLayer;
import org.brain4j.core.layer.impl.InputLayer;
import org.brain4j.core.model.impl.Sequential;
import org.brain4j.datasets.Datasets;
import org.brain4j.datasets.api.Dataset;
import org.brain4j.math.activation.impl.ReLU;
import org.brain4j.math.activation.impl.Softmax;
import org.brain4j.math.loss.impl.CrossEntropy;
import org.brain4j.core.model.ModelSpecs;
import org.brain4j.core.monitor.Monitor;
import org.brain4j.core.monitor.impl.EvalMonitor;
import org.brain4j.core.training.Trainer;
import org.brain4j.core.training.TrainingConfig;
import org.brain4j.core.training.optimizer.impl.AdamW;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.tensor.Shape;

import java.io.IOException;
import java.util.List;

public class TestMNIST {
    
    public static void main(String[] args) throws IOException {
        new TestMNIST().start();
    }
    
    private void start() throws IOException {
        Dataset dataset = Datasets.mnist(true, 128, 0.8);

        ListDataSource trainSource = dataset.train();
        ListDataSource testSource = dataset.test();

        ModelSpecs specs = getMLPSpecs();
        Sequential model = specs.compile(42);
        model.summary(); // prints a summary of the architecture on the console
        
        TrainingConfig config = TrainingConfig.of(
            new CrossEntropy(),
            new AdamW(0.01)
        );
        
        List<Monitor> monitors = List.of(
            new EvalMonitor(testSource, 5)
        );
        
        Trainer trainer = Trainer.of(model, config, monitors);
        trainer.fit(trainSource, 50);
    }

    private ModelSpecs getMLPSpecs() {
        return ModelSpecs.of(
            new InputLayer(Shape.of(28 * 28)),
            new DenseLayer(128, new ReLU()),
            new DenseLayer(64, new ReLU()),
            new DenseLayer(10, new Softmax())
        );
    }
}