package conv;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.convolution.Kernel;
import net.echo.brain4j.convolution.pooling.PoolingType;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.layer.impl.convolution.ConvLayer;
import net.echo.brain4j.layer.impl.convolution.FlattenLayer;
import net.echo.brain4j.layer.impl.convolution.InputLayer;
import net.echo.brain4j.layer.impl.convolution.PoolingLayer;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.impl.Sequential;
import net.echo.brain4j.model.initialization.WeightInit;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.optimizers.impl.Adam;
import net.echo.brain4j.training.optimizers.impl.AdamW;
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
        model.fit(dataSet);

        for (int i = 0; i < 1; i++) {
            long start = System.nanoTime();
            model.fit(dataSet);
            double took = (System.nanoTime() - start) / 1e6;

            System.out.println("Took " + took + " ms with " + dataSet.size() + " samples");

            double loss = model.evaluate(dataSet);
            System.out.println("Loss: " + loss);
        }

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
                new ConvLayer(20, 3, 3, Activations.MISH),
                new PoolingLayer(PoolingType.MAX, 2, 2, 2),

                // #2 convolutional block
                new ConvLayer(16, 5, 5, Activations.MISH),

                new PoolingLayer(PoolingType.MAX, 2, 2, 2),

                // Flattens the feature map to a 1D vector
                new FlattenLayer(25), // You must find the right size by trial and error

                // Classifiers
                new DenseLayer(32, Activations.MISH),
                new DenseLayer(10, Activations.SOFTMAX)
        );

        return model.compile(WeightInit.HE, LossFunctions.CROSS_ENTROPY, new AdamW(0.001), new StochasticUpdater());
    }

    private DataSet<DataRow> getDataSet() throws IOException {
        DataSet<DataRow> set = new DataSet<>();

        List<String> lines = FileUtils.readLines(new File("dataset.csv"), "UTF-8");

        int max = 1, i = 0;

        for (String line : lines) {
            i++;
            String[] parts = line.split(",");
            double[] inputs = Arrays.stream(parts, 1, parts.length).mapToDouble(x -> Double.parseDouble(x) / 255).toArray();

            Vector output = new Vector(10);

            int value = Integer.parseInt(parts[0]);
            output.set(value, 1);

            set.getData().add(new DataRow(Vector.of(inputs), output));

            if (i >= max) {
                break;
            }
        }

        return set;
    }
}
