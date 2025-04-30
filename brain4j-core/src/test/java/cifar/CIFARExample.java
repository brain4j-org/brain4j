package cifar;

import net.echo.brain4j.Brain4J;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.loss.Loss;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.impl.Sequential;
import net.echo.brain4j.training.evaluation.EvaluationResult;
import net.echo.brain4j.training.optimizer.impl.AdamW;
import net.echo.math.activation.Activations;
import net.echo.math.data.ListDataSource;
import net.echo.math.data.Sample;
import net.echo.math.tensor.Tensor;
import net.echo.math.tensor.Tensors;

import java.io.File;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

public class CIFARExample {

    public static void main(String[] args) throws Exception {
        new CIFARExample().start();
    }

    public void start() throws Exception {
        Brain4J.setLogging(true);

        // You can find the dataset on Kaggle
        // https://www.kaggle.com/datasets/fedesoriano/cifar10-python-in-csv
        ListDataSource trainSource = getDataSource("cifar_train.csv");

        System.out.println("Loaded " + trainSource.size() + " samples");

        Model model = new Sequential(
                new DenseLayer(3072, Activations.LINEAR),
                new DenseLayer(128, Activations.RELU),
                new DenseLayer(64, Activations.RELU),
                new DenseLayer(10, Activations.SOFTMAX)
        );

        model.load("cifar.b4j");
//        model.compile(Loss.CROSS_ENTROPY, new AdamW(0.01));

        System.out.println(model.summary());

        long start = System.nanoTime();
        model.fit(trainSource, 50, 1);
        double took = (System.nanoTime() - start) / 1e6;

        System.out.printf("Took %.4f milliseconds to complete the training%n", took);

        EvaluationResult result = model.evaluate(trainSource);
        System.out.println(result.confusionMatrix());

        model.save("cifar.b4j");
    }

    public ListDataSource getDataSource(String fileName) throws Exception {
        List<Sample> samples = new ArrayList<>();
        List<String> lines = Files.readAllLines(new File(fileName).toPath());

        for (int i = 1; i < lines.size(); i++) {
            String line = lines.get(i);
            String[] parts = line.split(",");

            float[] inputs = new float[parts.length - 1];

            for (int j = 0; j < inputs.length - 1; j++) {
                inputs[j] = Float.parseFloat(parts[j]);
            }

            Tensor input = Tensors.vector(inputs);
            Tensor output = Tensors.create(10);

            int label = Integer.parseInt(parts[parts.length - 1]);

            output.set(1, label);
            samples.add(new Sample(input, output));
        }

        return new ListDataSource(samples, true, 128);
    }
}
