package mnist;

import com.google.common.io.Files;
import net.echo.brain4j.Brain4J;
import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.adapters.ModernAdapter;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.loss.Loss;
import net.echo.brain4j.model.impl.Sequential;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.evaluation.EvaluationResult;
import net.echo.brain4j.training.optimizer.impl.AdamW;
import net.echo.math4j.DataSet;
import net.echo.math4j.math.tensor.TensorFactory;

import java.io.File;
import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class MNISTTest {

    public static void main(String[] args) throws Exception {
        new MNISTTest().testMNIST();
    }

    private void testMNIST() throws Exception {
        Brain4J.setLogging(true);

        var dataSet = getDataSet();
        var model = new Sequential(
                new DenseLayer(784, Activations.LINEAR),
                new DenseLayer(64, Activations.SIGMOID),
                new DenseLayer(10, Activations.SOFTMAX)
        );

        model.compile(Loss.CROSS_ENTROPY, new AdamW(0.01));

        if (new File("mnist.b4j").exists()) {
            System.out.println("Loading existing mnist.b4j model...");
            model.load("mnist.b4j");
        }

        System.out.println(model.summary());

        // model.fit(dataSet, 150, 10);

        EvaluationResult result = model.evaluate(dataSet);

        System.out.println(result.confusionMatrix());

        assertTrue(result.loss() < 0.1, "Loss is too high! " + result.loss());
        assertTrue(result.accuracy() > 0.90, "Accuracy is too low! " + result.accuracy());

        ModernAdapter.serialize("mnist", model);
    }

    private DataSet<DataRow> getDataSet() throws Exception {
        var dataSet = new DataSet<DataRow>();

        var file = new File("dataset.csv");
        var lines = Files.readLines(file, StandardCharsets.UTF_8);

        for (var line : lines) {
            var parts = line.split(",");

            var inputs = new float[parts.length - 1];

            for (int i = 0; i < inputs.length - 1; i++) {
                inputs[i] = Float.parseFloat(parts[i + 1]);
            }

            var input = TensorFactory.vector(inputs);
            var output = TensorFactory.create(10);

            int value = Integer.parseInt(parts[0]);
            output.set(1, value);

            dataSet.getData().add(new DataRow(input, output));
        }

        return dataSet;
    }
}
