package mnist;

import org.brain4j.core.Brain4J;
import org.brain4j.core.layer.impl.DenseLayer;
import org.brain4j.core.loss.Loss;
import org.brain4j.core.model.Model;
import org.brain4j.core.model.impl.Sequential;
import org.brain4j.core.training.evaluation.EvaluationResult;
import org.brain4j.core.training.optimizer.impl.AdamW;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.data.Sample;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

public class MNISTExample {

    public static void main(String[] args) throws Exception {
        new MNISTExample().start();
    }

    public void start() throws Exception {
        Brain4J.setLogging(true);

        // You can find the dataset on Kaggle
        // https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
        ListDataSource trainSource = getDataSource("mnist_train.csv");
        ListDataSource testSource = getDataSource("mnist_test.csv");

        Model model = new Sequential(
                new DenseLayer(784, Activations.LINEAR),
                new DenseLayer(128, Activations.RELU),
                new DenseLayer(64, Activations.RELU),
                new DenseLayer(10, Activations.SOFTMAX)
        );

        model.compile(Loss.CROSS_ENTROPY, new AdamW(0.01));
        System.out.println(model.summary());

        // model.load("mnist.b4j"); If you want to load the pre-trained model

        long start = System.nanoTime();
        model.fit(trainSource, testSource, 50, 10);
        double took = (System.nanoTime() - start) / 1e6;

        model.save("mnist.b4j");
        System.out.printf("Total time: %.4f ms | Average per epoch: %.4f ms %n", took, took / 50);

        EvaluationResult result = model.evaluate(testSource);
        System.out.println(result.results());
    }

    public ListDataSource getDataSource(String fileName) throws IOException {
        List<Sample> samples = new ArrayList<>();
        List<String> lines = Files.readAllLines(new File(fileName).toPath());

        for (int i = 1; i < lines.size(); i++) {
            String line = lines.get(i);
            String[] parts = line.split(",");

            float[] inputs = new float[parts.length - 1];

            for (int j = 1; j < inputs.length; j++) {
                inputs[j] = Float.parseFloat(parts[j]) / 255;
            }

            Tensor input = Tensors.vector(inputs);
            Tensor output = Tensors.create(10);

            output.set(1, Integer.parseInt(parts[0]));
            samples.add(new Sample(input, output));
        }

        return new ListDataSource(samples, true, 128);
    }
}
