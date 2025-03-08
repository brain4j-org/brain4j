package conv;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.layer.impl.convolution.ConvLayer;
import net.echo.brain4j.layer.impl.convolution.FlattenLayer;
import net.echo.brain4j.layer.impl.convolution.InputLayer;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.impl.Sequential;
import net.echo.brain4j.model.initialization.WeightInit;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.optimizers.impl.Adam;
import net.echo.brain4j.training.techniques.SmartTrainer;
import net.echo.brain4j.training.techniques.TrainListener;
import net.echo.brain4j.training.updater.impl.StochasticUpdater;
import net.echo.brain4j.utils.DataSet;
import net.echo.brain4j.utils.MLUtils;
import net.echo.brain4j.utils.Vector;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class ConvExample {

    public static void main(String[] args) throws IOException {
        ConvExample example = new ConvExample();
        example.start();
    }

    private void start() throws IOException {
        Sequential model = getModel();
        DataSet<DataRow> dataSet = getDataSet();

        System.out.println(model.getStats());

        SmartTrainer trainer = new SmartTrainer(1, 1);
        trainer.addListener(new TrainListener<DataRow>() {
            @Override
            public void onEvaluated(DataSet<DataRow> dataSet, int epoch, double loss, long took) {
                System.out.println("Epoch #" + epoch + " Loss: " + loss);
            }
        });
        trainer.startFor(model, dataSet, 100, 0.01);

        model.save("mnist-conv.json");

        int incorrect = 0;

        for (DataRow row : dataSet) {
            Vector input = row.inputs();
            Vector output = row.outputs();
            Vector prediction = model.predict(input);

            int expected = MLUtils.indexOfMaxValue(output) + 1;
            int predicted = MLUtils.indexOfMaxValue(prediction) + 1;

            if (expected != predicted) {
                System.out.println("Expected: " + expected + ", Prediction: " + predicted);
                incorrect++;
            }
        }

        int correct = dataSet.size() - incorrect;
        System.out.println("Correct: " + correct + ", Incorrect: " + incorrect);
    }

    private Sequential getModel() {
        Sequential model = new Sequential(
                // Input layer, necessary when using CNNs
                new InputLayer(28, 28),

                // #1 convolutional block
                new ConvLayer(32, 3, 3, Activations.MISH),

                // #2 convolutional block
                new ConvLayer(32, 5, 5, Activations.MISH),

                // Flattens the feature map to a 1D vector
                new FlattenLayer(484), // You must find the right size by trial and error

                // Classifiers
                new DenseLayer(32, Activations.MISH),
                new DenseLayer(10, Activations.SOFTMAX)
        );

        return model.compile(LossFunctions.CROSS_ENTROPY, new Adam(0.005));
    }

    private DataSet<DataRow> getDataSet() throws IOException {
        DataSet<DataRow> dataSet = new DataSet<>();
        List<String> lines = FileUtils.readLines(new File("dataset.csv"), "UTF-8");

        for (int j = 0; j < 150 * 2; j++) {
            String line = lines.get(j);
            String[] parts = line.split(",");
            double[] inputs = Arrays.stream(parts, 1, parts.length).mapToDouble(x -> Double.parseDouble(x) / 255).toArray();

            Vector output = new Vector(10);

            int value = Integer.parseInt(parts[0]);
            output.set(value, 1);

            dataSet.getData().add(new DataRow(Vector.of(inputs), output));
        }

        return dataSet;
    }
}
