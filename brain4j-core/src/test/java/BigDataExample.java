import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.layer.impl.DropoutLayer;
import net.echo.brain4j.loss.Loss;
import net.echo.brain4j.model.impl.Sequential;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.evaluation.EvaluationResult;
import net.echo.brain4j.training.optimizer.impl.AdamW;
import net.echo.math4j.DataSet;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;

public class BigDataExample {

    public static void main(String[] args) {
        new BigDataExample().start();
    }

    private void start() {
        Sequential model = getModel();
        DataSet<DataRow> dataSet = getDataSet(300_000);

        System.out.printf("Loaded dataset with %s samples %n", dataSet.size());
        System.out.println(model.summary());

        EvaluationResult result = model.evaluate(dataSet);
        System.out.println(result.confusionMatrix());

        long start = System.nanoTime();
        model.fit(dataSet);
        long took = System.nanoTime() - start;

        result = model.evaluate(dataSet);
        System.out.println(result.confusionMatrix());
        System.out.printf("Took %s ms to complete one epoch.", took / 1e6);
    }

    private Sequential getModel() {
        Sequential model = new Sequential(
                new DenseLayer(13, Activations.LINEAR),
                new DenseLayer(256, Activations.RELU),
                new DropoutLayer(0.2),
                new DenseLayer(256, Activations.RELU),
                new DropoutLayer(0.2),
                new DenseLayer(256, Activations.RELU),
                new DropoutLayer(0.2),
                new DenseLayer(1, Activations.SIGMOID)
        );

        return model.compile(Loss.BINARY_CROSS_ENTROPY, new AdamW(0.001));
    }

    private DataSet<DataRow> getDataSet(int samples) {
        DataSet<DataRow> dataSet = new DataSet<>();

        for (int i = 0; i < samples; i++) {
            Tensor input = TensorFactory.random(13);
            Tensor output = TensorFactory.random(1);

            dataSet.add(new DataRow(input, output));
        }

        return dataSet;
    }
}
