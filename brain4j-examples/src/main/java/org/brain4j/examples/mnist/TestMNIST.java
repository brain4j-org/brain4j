package org.brain4j.examples.mnist;

import org.brain4j.core.Brain4J;
import org.brain4j.core.importing.ModelZoo;
import org.brain4j.core.layer.impl.DenseLayer;
import org.brain4j.core.layer.impl.NormLayer;
import org.brain4j.core.layer.impl.convolutional.ConvLayer;
import org.brain4j.core.layer.impl.utility.ActivationLayer;
import org.brain4j.core.layer.impl.utility.InputLayer;
import org.brain4j.core.layer.impl.utility.ReshapeLayer;
import org.brain4j.math.loss.impl.CrossEntropy;
import org.brain4j.core.model.Model;
import org.brain4j.core.model.ModelBlock;
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
import org.brain4j.math.tensor.Tensor;

import java.io.File;
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
        SiliconDevice device = Brain4J.firstDevice();
        
        ListDataSource trainSource = getSource("mnist/mnist-train.csv");
        ListDataSource testSource = getSource("mnist/mnist-test.csv");
        
        ModelSpecs specs = getCNNSpecs();
        Model model = specs.compile(42);
        model.summary(); // prints a summary of the architecture on the console
        
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
            new EvalMonitor(testSource, 10)
        );
        
        Trainer trainer = Trainer.of(model, config, monitors);
        trainer.fit(trainSource, 50);
        
        ModelZoo.saveModel(model, new File("mnist-100k.csv"));
    }
    
    private ModelSpecs getCNNSpecs() {
        return ModelSpecs.of(
            new InputLayer(28 * 28),
            new ReshapeLayer(1, 28, 28),
            new ConvLayer(1, 16, 3, 3, 1, Activations.LEAKY_RELU), // 16x26x26
            new ConvLayer(16, 32, 3, 3, 2, Activations.LEAKY_RELU), // 32x12x12
            new ConvLayer(32, 64, 3, 3, 2, Activations.LEAKY_RELU), // 64x5x5
            new ReshapeLayer(64 * 5 * 5),
            new DenseLayer(10, Activations.SOFTMAX)
        );
    }
    
    private ModelSpecs getMLPSpecs() {
        return ModelSpecs.of(
            new InputLayer(28 * 28),
            new DenseLayer(128, Activations.RELU),
            new DenseLayer(64, Activations.RELU),
            new DenseLayer(10, Activations.SOFTMAX)
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
        
        return new ListDataSource(samples, false, 128);
    }
}