import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.convolution.impl.ConvLayer;
import net.echo.brain4j.convolution.impl.FlattenLayer;
import net.echo.brain4j.convolution.impl.InputLayer;
import net.echo.brain4j.convolution.impl.PoolingLayer;
import net.echo.brain4j.convolution.pooling.PoolingType;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.initialization.WeightInit;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.data.DataSet;
import net.echo.brain4j.training.optimizers.impl.Adam;
import net.echo.brain4j.training.updater.impl.StochasticUpdater;
import net.echo.brain4j.utils.Vector;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class ConvExample {

    public static void main(String[] args) {
        ConvExample example = new ConvExample();
        example.start();
    }

    private void start() {
        Model model = getModel();
        DataSet dataSet = getDataSet();

        System.out.println(model.getStats());
        long start = System.nanoTime();
        double loss = model.evaluate(dataSet);
        double took = (System.nanoTime() - start) / 1e6;

        System.out.println(took + " -> " + loss);
    }

    private Model getModel() {
        Model model = new Model(
                // Input layer, necessary when using CNNs
                new InputLayer(28, 28),

                // #1 convolutional block
                new ConvLayer(32, 3, 3, Activations.RELU),
                new PoolingLayer(PoolingType.MAX, 2, 2, 2),

                // #2 convolutional block
                new ConvLayer(64, 5, 5, Activations.RELU),
                new PoolingLayer(PoolingType.MAX, 2, 2, 2),

                // Flattens the feature map to a 1D vector
                new FlattenLayer(16), // You must find the right size by trial and error

                // Classifiers
                new DenseLayer(32, Activations.RELU),
                new DenseLayer(10, Activations.SOFTMAX)
        );

        return model.compile(WeightInit.HE, LossFunctions.CROSS_ENTROPY, new Adam(0.1), new StochasticUpdater());
    }

    private DataSet getDataSet() {
        DataSet set = new DataSet();

        try {
            List<String> lines = FileUtils.readLines(new File("dataset.csv"), "UTF-8");

            for (String line : lines) {
                String[] parts = line.split(",");
                double[] inputs = Arrays.stream(parts, 1, parts.length).mapToDouble(x -> Double.parseDouble(x) / 255).toArray();

                Vector output = new Vector(10);

                int value = Integer.parseInt(parts[0]);
                output.set(value, 1);

                set.getData().add(new DataRow(Vector.of(inputs), output));
            }
        } catch (IOException e) {
            throw new RuntimeException("Error reading dataset: " + e.getMessage(), e);
        }

        return set;
    }
}
