package mnist;

import com.google.common.io.CharStreams;
import com.google.common.io.Files;
import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.adapters.ModernAdapter;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.impl.Sequential;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.evaluation.EvaluationResult;
import net.echo.brain4j.training.optimizers.impl.Adam;
import net.echo.brain4j.training.techniques.EpochListener;
import net.echo.brain4j.training.techniques.SmartTrainer;
import net.echo.math4j.DataSet;
import net.echo.math4j.math.tensor.TensorFactory;
import net.echo.math4j.math.vector.Vector;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;

public class MNISTClassifier {

    public static void main(String[] args) throws Exception {
        TensorFactory.useGPUIfAvailable();

        DataSet<DataRow> dataSet = getData();
        Sequential model = getModel();

        System.out.println(model.getStats());

        train(model, dataSet);
        evaluateModel(model, dataSet);
    }

    public static void evaluateModel(Sequential model, DataSet<DataRow> dataSet) {
        model.reloadWeights();

        EvaluationResult result = model.evaluate(dataSet);
        double loss = model.loss(dataSet);

        System.out.println(result.confusionMatrix());
        System.out.println("Loss: " + loss);
    }

    public static void train(Sequential model, DataSet<DataRow> set) throws Exception {
        SmartTrainer<DataRow> trainer = new SmartTrainer<>(0.7, 1);

        trainer.addListener(new EpochListener<>());
        trainer.startFor(model, set, 50, 0.01);

        ModernAdapter.serialize("mnist", model);

        System.out.println("Took: " + trainer.getTook() / 1e6 + " ms");
    }

    public static Sequential getModel() {
        Sequential model = new Sequential(
                new DenseLayer(784, Activations.LINEAR),
                new DenseLayer(32, Activations.SIGMOID),
                new DenseLayer(10, Activations.SOFTMAX)
        );

        return model.compile(LossFunctions.CROSS_ENTROPY, new Adam(0.01));
    }

    public static DataSet<DataRow> getData() throws IOException {
        DataSet<DataRow> dataSet = new DataSet<>();

        File file = new File("dataset.csv");
        List<String> lines = Files.readLines(file, StandardCharsets.UTF_8);

        for (String line : lines) {
            String[] parts = line.split(",");
            double[] inputs = Arrays.stream(parts, 1, parts.length).mapToDouble(Double::parseDouble).toArray();

            Vector output = new Vector(10);

            int value = Integer.parseInt(parts[0]);
            output.set(value, 1);

            dataSet.getData().add(new DataRow(Vector.of(inputs), output));
        }

        return dataSet;
    }
}