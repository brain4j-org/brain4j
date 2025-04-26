package mnist;

import net.echo.brain4j.Brain4J;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.loss.Loss;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.impl.Sequential;
import net.echo.brain4j.training.optimizer.impl.AdamW;
import net.echo.math.activation.Activations;
import net.echo.math.data.ListDataSource;
import net.echo.math.data.Sample;
import net.echo.math.tensor.Tensor;
import net.echo.math.tensor.Tensors;

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

        model.fit(trainSource, testSource, 50, 10);
        model.save("mnist.b4j");
    }

    public ListDataSource getDataSource(String fileName) throws IOException {
        List<Sample> samples = new ArrayList<>();
        List<String> lines = Files.readAllLines(new File(fileName).toPath());

        for (int i = 1; i < lines.size(); i++) {
            String line = lines.get(i);
            String[] parts = line.split(",");

            float[] inputs = new float[parts.length - 1];

            for (int j = 1; j < inputs.length; j++) {
                inputs[j] = Float.parseFloat(parts[j]);
            }

            Tensor input = Tensors.vector(inputs);
            Tensor output = Tensors.create(10);

            output.set(1, Integer.parseInt(parts[0]));
            samples.add(new Sample(input, output));
        }

        return new ListDataSource(samples, true, 128);
    }
}
