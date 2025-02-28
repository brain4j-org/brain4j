import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.initialization.WeightInit;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.optimizers.impl.Adam;
import net.echo.brain4j.training.updater.impl.StochasticUpdater;
import net.echo.brain4j.utils.DataSet;
import net.echo.brain4j.utils.Vector;

public class XorExample {

    public static void main(String[] args) {
        XorExample example = new XorExample();
        example.start();
    }

    private void start() {
        Model model = getModel();
        DataSet<DataRow> dataSet = getDataSet();

        System.out.println(model.getStats());

        // Fit the model for 1000 epoches
        for (int i = 0; i < 1000; i++) {
            model.fit(dataSet);
        }

        for (DataRow row : dataSet) {
            Vector prediction = model.predict(row.inputs());

            System.out.println("Expected: " + row.outputs() + " -> Predicted: " + prediction.toString("%.3f"));
        }
    }

    private Model getModel() {
        Model model = new Model(
                new DenseLayer(2, Activations.LINEAR), // 2 Input neurons
                new DenseLayer(32, Activations.MISH), // 32 Hidden neurons
                new DenseLayer(32, Activations.MISH), // 32 Hidden neurons
                new DenseLayer(1, Activations.SIGMOID) // 1 Output neuron for classification
        );

        return model.compile(WeightInit.HE, LossFunctions.BINARY_CROSS_ENTROPY, new Adam(0.1), new StochasticUpdater());
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
