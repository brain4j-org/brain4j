package org.brain4j.examples.xor;

import org.brain4j.core.layer.newimpl.DenseLayer;
import org.brain4j.core.layer.newimpl.InputLayer;
import org.brain4j.core.model.impl.Sequential;
import org.brain4j.dashboard.BrainDashboard;
import org.brain4j.math.activation.impl.ReLU;
import org.brain4j.math.activation.impl.Sigmoid;
import org.brain4j.math.loss.impl.BinaryCrossEntropy;
import org.brain4j.core.model.ModelSpecs;
import org.brain4j.core.monitor.Monitor;
import org.brain4j.core.monitor.impl.EvalMonitor;
import org.brain4j.core.monitor.impl.ProgressMonitor;
import org.brain4j.core.training.Trainer;
import org.brain4j.core.training.TrainingConfig;
import org.brain4j.core.training.optimizer.impl.AdamW;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.data.Sample;
import org.brain4j.math.gpu.silicon.SiliconDevice;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

public class XorRegression {
    public static void main(String[] args) {
        List<Sample> samples = getSamples();
        
        ListDataSource dataSource = new ListDataSource(samples, false, 1);
        ModelSpecs specs = ModelSpecs.of(
            new InputLayer(Shape.of(2)),
            new DenseLayer(16, new ReLU()),
            new DenseLayer(16, new ReLU()),
            new DenseLayer(1, new Sigmoid())
        );
        
        Sequential model = specs.compile(42);
        SiliconDevice device = null;

        model.summary();
        
        if (device != null) {
            System.out.println("Using device " + device.getName());
//            model = model.fork(device);
//            dataSource = dataSource.to(device);
        }
        
        TrainingConfig config = TrainingConfig.of(
            new BinaryCrossEntropy(),
            new AdamW(0.1)
        );
        List<Monitor> monitors = List.of(
            new ProgressMonitor(),
            new EvalMonitor(dataSource, 10)
        );

        Trainer trainer = Trainer.of(model, config, monitors);
        // trainer.fit(dataSource, 50);

        BrainDashboard dashboard = new BrainDashboard(model, trainer);
        dashboard.launch(dataSource, dataSource, 50, 8080);
    }
    
    private static List<Sample> getSamples() {
        List<Sample> samples = new ArrayList<>();
        
        for (int x = 0; x <= 1; x++) {
            for (int y = 0; y <= 1; y++) {
                Tensor input = Tensors.vector(x, y);
                Tensor label = Tensors.vector(x ^ y);
                
                samples.add(new Sample(input, label));
            }
        }
        return samples;
    }
}
