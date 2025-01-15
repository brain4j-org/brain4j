import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.initialization.WeightInit;
import net.echo.brain4j.training.data.DataSet;
import net.echo.brain4j.training.optimizers.impl.AdamW;
import net.echo.brain4j.training.updater.impl.NormalUpdater;
import net.echo.brain4j.utils.Vector;

public class ParallelizationTest {

    public static void main(String[] args) {
        Model model = getModel();
        DataSet set = getDataSet();

        System.out.println(model.getStats());
        set.partition(1);

        int i = 0;
        double error = Double.MAX_VALUE;
        double lastError = Double.MAX_VALUE;

        long start = System.nanoTime();

        do {
            model.fit(set);

            if (i++ % 100 == 0) {
                long end = System.nanoTime();
                error = model.evaluate(set);

                System.out.println("Epoch " + i + " took " + (end - start) / 1000000 + "ms and has " + error + " loss");
                start = end;

                if (lastError != Double.MAX_VALUE) {
                    double difference = error - lastError;

                    if (difference > 0 || Math.abs(difference) < 1e-5) {
                        System.out.println("Loss increased, decreasing learning rate.");

                        double learningRate = model.getOptimizer().getLearningRate();

                        model.getOptimizer().setLearningRate(learningRate * 0.6);
                        System.out.println("Old learning rate: " + learningRate + " | New learning rate: " + model.getOptimizer().getLearningRate());
                    }
                }

                lastError = error;
            }
        } while (error > 0.01 && i < 30000);

        for (int j = 0; j < 30; j++) {
            Vector input = Vector.of(j);

            Vector output = Vector.of(j / 30.0);
            Vector prediction = model.predict(input);

            double difference = output.distance(prediction);

            System.out.println(prediction + " -> " + output + " | " + difference);
        }
    }

    private static Model getModel() {
        Model model = new Model(
                new DenseLayer(1, Activations.LINEAR),
                new DenseLayer(32, Activations.RELU),
                new DenseLayer(16, Activations.RELU),
                new DenseLayer(1, Activations.SIGMOID)
        );

        model.setSeed(123);
        model.compile(
                WeightInit.NORMAL_XAVIER,
                LossFunctions.MEAN_SQUARED_ERROR,
                new AdamW(0.001),
                new NormalUpdater()
        );

        return model;
    }

    private static DataSet getDataSet() {
        DataSet set = new DataSet();

        for (int i = 0; i < 30; i++) {
            Vector input = Vector.of(i);
            Vector output = Vector.of(i / 30.0);

            set.add(input, output);
        }

        return set;
    }
}
