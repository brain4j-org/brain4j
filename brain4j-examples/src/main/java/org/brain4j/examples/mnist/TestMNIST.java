package org.brain4j.examples.mnist;

import org.brain4j.core.Brain4J;
import org.brain4j.core.layer.impl.DenseLayer;
import org.brain4j.core.layer.impl.InputLayer;
import org.brain4j.core.model.impl.Sequential;
import org.brain4j.core.training.wrappers.EvaluationResult;
import org.brain4j.datasets.Datasets;
import org.brain4j.datasets.api.Dataset;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.impl.ReLU;
import org.brain4j.math.activation.impl.Softmax;
import org.brain4j.math.data.Sample;
import org.brain4j.math.gpu.device.Device;
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
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class TestMNIST {
    
    static void main() throws IOException {
        new TestMNIST().start();
    }
    
    private void start() {
        Dataset dataset;
        try {
            dataset = Datasets.mnist(true, 128, 0.8);
        } catch (Exception e) {
            System.err.println("Failed to load real MNIST, using dummy dataset: " + e.getMessage());
            dataset = createDummyMnist(128, 0.8);
        }

        Device device = Brain4J.firstDevice();

        ListDataSource trainSource = dataset.train().to(device);
        ListDataSource testSource = dataset.test().to(device);

        ModelSpecs specs = getMLPSpecs();
        Sequential model = specs.compile(42).fork(device);
        model.summary(); // prints a summary of the architecture on the console

        TrainingConfig config = TrainingConfig.of(
            new CrossEntropy(),
            new AdamW(0.01)
        );

        List<Monitor> monitors = List.of(
            new EvalMonitor(testSource, 10)
        );

        Trainer trainer = Trainer.of(model, config, monitors);
        trainer.fit(trainSource, 50);

        EvaluationResult result = model.evaluate(testSource, new CrossEntropy());
        System.out.println(result.results());
    }

    private ModelSpecs getMLPSpecs() {
        return ModelSpecs.of(
            new InputLayer(Shape.of(28 * 28)),
            new DenseLayer(128, new ReLU()),
            new DenseLayer(64, new ReLU()),
            new DenseLayer(10, new Softmax())
        );
    }

    private Dataset createDummyMnist(int batchSize, double split) {
        int total = 2000;
        int trainSize = (int) (total * split);

        Random rnd = new Random(42);
        List<Sample> all = new ArrayList<>();

        for (int i = 0; i < total; i++) {
            float[] img = new float[28 * 28];

            for (int j = 0; j < img.length; j++)
                img[j] = rnd.nextFloat();

            int label = rnd.nextInt(10);
            float[] oneHot = new float[10];
            oneHot[label] = 1f;
            all.add(new Sample(
                Tensors.vector(img),
                Tensors.vector(oneHot)
            ));
        }

        var train = new ListDataSource(all.subList(0, trainSize), true, batchSize);
        var test = new ListDataSource(all.subList(trainSize, total), true, batchSize);

        return new Dataset(split, train, test);
    }
}