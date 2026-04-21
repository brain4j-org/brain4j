package org.brain4j.examples.mnist;

import org.brain4j.core.Brain4J;
import org.brain4j.core.layer.impl.ConvLayer;
import org.brain4j.core.layer.impl.DenseLayer;
import org.brain4j.core.layer.impl.InputLayer;
import org.brain4j.core.layer.impl.ReshapeLayer;
import org.brain4j.core.model.impl.Sequential;
import org.brain4j.dashboard.BrainDashboard;
import org.brain4j.math.activation.impl.LeakyReLU;
import org.brain4j.math.activation.impl.ReLU;
import org.brain4j.math.activation.impl.Softmax;
import org.brain4j.math.loss.impl.CrossEntropy;
import org.brain4j.core.model.ModelSpecs;
import org.brain4j.core.monitor.Monitor;
import org.brain4j.core.monitor.impl.EvalMonitor;
import org.brain4j.core.monitor.impl.ProgressMonitor;
import org.brain4j.core.training.Trainer;
import org.brain4j.core.training.TrainingConfig;
import org.brain4j.core.training.optimizer.impl.AdamW;
import org.brain4j.math.Tensors;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.data.Sample;
import org.brain4j.math.gpu.silicon.SiliconDevice;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class TestMNIST {
    
    public static void main(String[] args) throws IOException {
        new TestMNIST().start();
    }
    
    private void start() throws IOException {
        ListDataSource trainSource = getSource("mnist/mnist-train.csv");
        ListDataSource testSource = getSource("mnist/mnist-test.csv");
        
        ModelSpecs specs = getMLPSpecs();
        Sequential model = specs.compile(42);
        model.summary(); // prints a summary of the architecture on the console

        SiliconDevice device = Brain4J.firstDevice();
        if (device != null) {
            model = model.fork(device);
            trainSource = trainSource.to(device);
            testSource = testSource.to(device);
        }
        
        TrainingConfig config = TrainingConfig.of(
            new CrossEntropy(),
            new AdamW(0.01)
        );
        
        List<Monitor> monitors = List.of(
            new ProgressMonitor(),
            new EvalMonitor(testSource, 1)
        );
        
        Trainer trainer = Trainer.of(model, config, monitors);

        BrainDashboard dashboard = new BrainDashboard(model, trainer);
        dashboard.launch(trainSource, testSource, 50, 8080);
    }
    
    private ModelSpecs getCNNSpecs() {
        return ModelSpecs.of(
            new InputLayer(Shape.of(28 * 28)),
            new ReshapeLayer(Shape.of(1, 28, 28)),
            new ConvLayer(16, 3, 3, 1, new LeakyReLU()), // 16x26x26
            new ConvLayer(32, 3, 3, 2, new LeakyReLU()), // 32x12x12
            new ConvLayer(64, 3, 3, 2, new LeakyReLU()), // 64x5x5
            new ReshapeLayer(Shape.of(64 * 5 * 5)),
            new DenseLayer(10, new Softmax())
        );
    }
    
    private ModelSpecs getMLPSpecs() {
        return ModelSpecs.of(
            new InputLayer(Shape.of(28 * 28)),
            new DenseLayer(128, new ReLU()),
            new DenseLayer(64, new ReLU()),
            new DenseLayer(10, new Softmax())
        );
    }
    
    public ListDataSource getSource(String file) throws IOException {
        List<String> lines = Files.readAllLines(Paths.get(file));
        List<Sample> samples = new ArrayList<>();
        
        for (int i = 1; i < lines.size(); i++) {
            String line = lines.get(i);
            String[] tokens = line.split(",");
            
            int label = Integer.parseInt(tokens[0]);
            float[] values = new float[tokens.length - 1];
            
            for (int j = 0; j < values.length; j++) {
                values[j] = Float.parseFloat(tokens[j + 1]) / 255.0f;
            }
            
            Tensor input = Tensors.vector(values);
            Tensor output = Tensors.zeros(10).set(1, label);
            
            samples.add(new Sample(input, output));
        }
        
        return new ListDataSource(samples, true, 128);
    }
}