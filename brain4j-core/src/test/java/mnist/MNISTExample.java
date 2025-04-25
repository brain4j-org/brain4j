package mnist;

import com.google.common.io.Files;
import net.echo.brain4j.Brain4J;
import net.echo.brain4j.adapters.ModernAdapter;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.loss.Loss;
import net.echo.brain4j.model.impl.Sequential;
import net.echo.brain4j.training.evaluation.EvaluationResult;
import net.echo.brain4j.training.optimizer.impl.AdamW;
import net.echo.math.activation.Activations;
import net.echo.math.data.ListDataSource;
import net.echo.math.data.Sample;
import net.echo.math.tensor.Tensor;
import net.echo.math.tensor.Tensors;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

public class MNISTExample {

    public static void main(String[] args) throws Exception {
        new MNISTExample().start();
    }

    private void start() throws Exception {
        Brain4J.setLogging(true);

        ListDataSource source = getDataSet();
        Sequential model = new Sequential(
                new DenseLayer(784, Activations.LINEAR),
                new DenseLayer(64, Activations.SIGMOID),
                new DenseLayer(10, Activations.SOFTMAX)
        );

        model.compile(Loss.CROSS_ENTROPY, new AdamW(0.01));

        System.out.println(model.summary());

        long start = System.nanoTime();
        model.fit(source, 150, 10);
        double tookMillis = (System.nanoTime() - start) / 1e6;

        System.out.println("Took: " + tookMillis + " ms");

        EvaluationResult result = model.evaluate(source);
        System.out.println(result.confusionMatrix());

        ModernAdapter.serialize("mnist.b4j", model);
    }

    private ListDataSource getDataSet() throws Exception {
        List<Sample> samples = new ArrayList<>();

        File file = new File("dataset.csv");
        List<String> lines = Files.readLines(file, StandardCharsets.UTF_8);

        for (String line : lines) {
            String[] parts = line.split(",");

            float[] inputs = new float[parts.length - 1];

            for (int i = 0; i < inputs.length - 1; i++) {
                inputs[i] = Float.parseFloat(parts[i + 1]);
            }

            Tensor input = Tensors.vector(inputs);
            Tensor output = Tensors.create(10);

            int value = Integer.parseInt(parts[0]);
            output.set(1, value);

            samples.add(new Sample(input, output));
        }

        return new ListDataSource(samples, 100);
    }
}
