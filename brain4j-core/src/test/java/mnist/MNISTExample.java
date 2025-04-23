package mnist;

import com.google.common.io.Files;
import net.echo.brain4j.Brain4J;
import net.echo.brain4j.clipping.impl.HardClipper;
import net.echo.brain4j.clipping.impl.L2Clipper;
import net.echo.math.BrainUtils;
import net.echo.math.activation.Activations;
import net.echo.brain4j.adapters.ModernAdapter;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.loss.Loss;
import net.echo.brain4j.model.impl.Sequential;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.evaluation.EvaluationResult;
import net.echo.brain4j.training.optimizer.impl.AdamW;
import net.echo.math.DataSet;
import net.echo.math.tensor.Tensors;

import java.io.File;
import java.nio.charset.StandardCharsets;

public class MNISTExample {

    public static void main(String[] args) throws Exception {
        new MNISTExample().start();
    }

    private void start() throws Exception {
        Brain4J.setLogging(true);
        L2Clipper clipper = new L2Clipper(1);

        DataSet<DataRow> dataSet = getDataSet();
        Sequential model = new Sequential(
                new DenseLayer(784, Activations.LINEAR, clipper),
                new DenseLayer(64, Activations.SIGMOID, clipper),
                new DenseLayer(10, Activations.SOFTMAX, clipper)
        );

        model.compile(Loss.CROSS_ENTROPY, new AdamW(0.01));

        System.out.println(model.summary());
        model.fit(dataSet, 150, 10);

        EvaluationResult result = model.evaluate(dataSet);

        System.out.println(result.confusionMatrix());

        ModernAdapter.serialize("mnist.b4j", model);
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

            var input = Tensors.vector(inputs);
            var output = Tensors.create(10);

            int value = Integer.parseInt(parts[0]);
            output.set(1, value);

            dataSet.getData().add(new DataRow(input, output));
        }

        return dataSet;
    }
}
