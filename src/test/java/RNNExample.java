import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.impl.Sequential;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.optimizers.impl.Adam;
import net.echo.brain4j.utils.DataSet;
import net.echo.brain4j.utils.Vector;

public class RNNExample {

    public static void main(String[] args) {
        RNNExample example = new RNNExample();
        example.start();
    }

    public void start() {
        Sequential model = getModel();
        DataSet<DataRow> dataSet = getDataSet();

        for (int i = 0; i < 10000; i++) {
            model.fit(dataSet);

            if (i % 1000 == 0) {
                double loss = model.loss(dataSet);

                System.out.println(i + ". Loss: " + loss);
            }
        }

        for (DataRow row : dataSet) {
            Vector input = row.inputs();
            Vector output = row.outputs();

            Vector prediction = model.predict(input);
            System.out.println("Input: " + input + ", Output: " + output + ", Prediction: " + prediction);
        }
    }

    public Sequential getModel() {
        Sequential model = new Sequential(
                new DenseLayer(1, Activations.LINEAR),
                new DenseLayer(32, Activations.MISH),
                new DenseLayer(32, Activations.MISH),
                new DenseLayer(1, Activations.LINEAR)
        );

        return model.compile(LossFunctions.MEAN_SQUARED_ERROR, new Adam(0.01));
    }

    public DataSet<DataRow> getDataSet() {
        DataSet<DataRow> dataSet = new DataSet<>();

        for (int i = 1; i < 100; i++) {
            Vector input = Vector.of(i);
            Vector output = Vector.of(i + 1);

            dataSet.add(new DataRow(input, output));
        }

        return dataSet;
    }
}
