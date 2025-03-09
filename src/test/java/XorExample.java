import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.impl.Sequential;
import net.echo.brain4j.model.initialization.WeightInit;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.optimizers.impl.AdamW;
import net.echo.brain4j.training.updater.impl.StochasticUpdater;
import net.echo.brain4j.utils.DataSet;
import net.echo.brain4j.utils.Vector;

public class XorExample {

    public static void main(String[] args) {
        XorExample example = new XorExample();
        example.start();
    }

    private void start() {
        Sequential model = getModel();
        DataSet<DataRow> dataSet = getDataSet();

        System.out.println(model.getStats());

        long start = System.nanoTime();

        // Fit the model for 1000 epoches
        for (int i = 0; i < 100; i++) {
            model.fit(dataSet);
        }

        double took = (System.nanoTime() - start) / 1e6;

        System.out.println("Loss: " + model.loss(dataSet));
        System.out.println("Took: " + took + " ms");

        for (DataRow row : dataSet) {
            Vector prediction = model.predict(row.inputs());

            System.out.println("Expected: " + row.outputs() + " -> Predicted: " + prediction.toString("%.3f"));
        }
    }

    private Sequential getModel() {
        Sequential model = new Sequential(
                new DenseLayer(2, Activations.LINEAR), // 2 Input neurons
                new DenseLayer(32, Activations.MISH), // 32 Hidden neurons
                new DenseLayer(32, Activations.MISH), // 32 Hidden neurons
                new DenseLayer(1, Activations.SIGMOID) // 1 Output neuron for classification
        );

        return model.compile(WeightInit.HE, LossFunctions.BINARY_CROSS_ENTROPY, new AdamW(0.1), new StochasticUpdater());
    }

    private DataSet<DataRow> getDataSet() {
        DataSet<DataRow> set = new DataSet<>();

        for (int x = 0; x < 2; x++) {
            for (int y = 0; y < 2; y++) {
                int output = x ^ y;

                set.add(new DataRow(Vector.of(x, y), Vector.of(output)));
            }
        }

        return set;
    }
}
