package org.brain4j.examples.mnist;

import org.brain4j.core.Brain4J;
import org.brain4j.core.importing.ModelZoo;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.impl.DenseLayer;
import org.brain4j.core.layer.impl.NormLayer;
import org.brain4j.core.layer.impl.utility.ActivationLayer;
import org.brain4j.core.layer.impl.utility.InputLayer;
import org.brain4j.core.loss.impl.CrossEntropy;
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
        
        ModelSpecs specs = ModelSpecs.of(
            new InputLayer(28 * 28),
            new DenseLayer(128, Activations.RELU),
            new DenseLayer(64, Activations.RELU),
            new DenseLayer(10, Activations.SOFTMAX)
        );
        
        Model model = specs.compile(42);
        model.summary(); // prints a summary of the architecture on the console
        
        if (device != null) {
//            model = model.fork(device);
//            trainSource = trainSource.to(device);
//            testSource = testSource.to(device);
        }
        
        TrainingConfig config = TrainingConfig.of(
            new CrossEntropy(),
            new AdamW(0.01)
        );
        
        List<Monitor> monitors = List.of(
            new ProgressMonitor(),
            new EvalMonitor(testSource, 5)
        );
        
        Trainer trainer = Trainer.of(model, config, monitors);
        trainer.fit(trainSource, 50);
        
        ModelZoo.saveModel(model, new File("mnist-100k.csv"));
    }
    
    private ModelBlock denseNormActivation(int dimension) {
        return layers -> layers.addAll(
            List.of(
                new DenseLayer(dimension),
                new NormLayer(),
                new ActivationLayer(Activations.RELU)
            )
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