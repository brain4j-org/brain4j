import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.layer.impl.recurrent.RecurrentLayer;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.Model;
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
        Model model = getModel();
        DataSet<DataRow> dataSet = getDataSet();

        model.predict(Vector.of(3, 3));
//        for (int i = 0; i < 1000; i++) {
//            model.fit(dataSet);
//            double loss = model.evaluate(dataSet);
//
//            System.out.println("Loss: " + loss);
//        }

//        for (DataRow row : dataSet) {
//            Vector input = row.inputs();
//            Vector output = row.outputs();
//
//            Vector prediction = model.predict(input);
//            System.out.println("Input: " + input + ", Output: " + output + ", Prediction: " + prediction);
//        }
    }

    public Model getModel() {
        Model model = new Model(
                new DenseLayer(2, Activations.LINEAR),
                new RecurrentLayer(4, Activations.TANH),
                new DenseLayer(1, Activations.LINEAR)
        );

        return model.compile(LossFunctions.MEAN_SQUARED_ERROR, new Adam(0.001));
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
