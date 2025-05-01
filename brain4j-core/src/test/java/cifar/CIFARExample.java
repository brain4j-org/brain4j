package cifar;

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

        model.compile(Loss.CROSS_ENTROPY, new AdamW(0.01));

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
