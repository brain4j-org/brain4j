import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.initialization.WeightInit;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.data.DataSet;
import net.echo.brain4j.training.optimizers.impl.Adam;
import net.echo.brain4j.training.optimizers.impl.AdamW;
import net.echo.brain4j.training.updater.impl.NormalUpdater;
import net.echo.brain4j.utils.Vector;

public class XorTest {

    public static void main(String[] args) {
        Model model = new Model(
                new DenseLayer(2, Activations.LINEAR),
                new DenseLayer(32, Activations.RELU),
                new DenseLayer(32, Activations.RELU),
                new DenseLayer(1, Activations.SIGMOID)
        );

        model.compile(
                WeightInit.HE,
                LossFunctions.BINARY_CROSS_ENTROPY,
                new Adam(0.1),
                new NormalUpdater()
        );

        System.out.println(model.getStats());

        DataRow first = new DataRow(Vector.of(0, 0), Vector.of(0));
        DataRow second = new DataRow(Vector.of(0, 1), Vector.of(1));
        DataRow third = new DataRow(Vector.of(1, 0), Vector.of(1));
        DataRow fourth = new DataRow(Vector.of(1, 1), Vector.of(0));

        DataSet training = new DataSet(first, second, third, fourth);
        training.partition(1);

        trainForBenchmark(model, training);
    }

    private static void trainForBenchmark(Model model, DataSet data) {
        double total = 0.0;

        long start = System.nanoTime();

        for (int i = 0; i < 5000; i++) {
            model.fit(data);
        }

        long end = System.nanoTime();
        double took = (end - start) / 1e6;

        double error = model.evaluate(data);
        double mean = took / 5000;

        System.out.println("Completed 5000 epoches in " + took + " ms with error: " + error + " and an average of " + mean + " ms per epoch");
    }

    private static void trainTillError(Model model, DataSet data) {
        double error;
        int epoches = 0;

        do {
            model.fit(data);

            error = model.evaluate(data);
            epoches++;

            System.out.println("Epoch " + epoches + " error: " + error);
        } while (error > 0.01);
    }
}
