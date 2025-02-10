import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.initialization.WeightInit;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.data.DataSet;
import net.echo.brain4j.training.optimizers.impl.AdamW;
import net.echo.brain4j.training.techniques.SmartTrainer;
import net.echo.brain4j.training.techniques.TrainListener;
import net.echo.brain4j.training.updater.impl.StochasticUpdater;
import net.echo.brain4j.utils.MLUtils;
import net.echo.brain4j.utils.Vector;

import java.util.concurrent.ThreadLocalRandom;

public class TransactionsTest {

    private static final int INPUT_DIMENSION = 30;

    public static void main(String[] args) {
        Model model = getModel();

        DataSet dataSet = new DataSet(getKarhu(), getPolar(), getGrim(), getVulcan(), getMatrix(), getIntave());

        SmartTrainer trainer = new SmartTrainer(1, 1_000);

        trainer.addListener(new TestListener());
        trainer.startFor(model, dataSet, 20000);

        for (DataRow row : dataSet) {
            Vector output = model.predict(row.inputs());

            AntiCheat predicted = MLUtils.findBestMatch(output, AntiCheat.class);
            AntiCheat actual = MLUtils.findBestMatch(row.outputs(), AntiCheat.class);

            System.out.println("Anticheat (Prediction, Actual): " + predicted + ", " + actual);
            System.out.println("Distribution: " + output.toString("%.2f"));
        }
    }

    public static Model getModel() {
        Model model = new Model(
                new DenseLayer(INPUT_DIMENSION, Activations.RELU),
                new DenseLayer(64, Activations.SIGMOID),
                new DenseLayer(32, Activations.SIGMOID),
                new DenseLayer(6, Activations.SOFTMAX)
        );

        model.compile(WeightInit.UNIFORM_XAVIER, LossFunctions.CROSS_ENTROPY, new AdamW(0.01), new StochasticUpdater());

        return model;
    }

    public static DataRow getKarhu() {
        Vector vector = new Vector(INPUT_DIMENSION);

        for (int i = 0; i < INPUT_DIMENSION; i++) {
            vector.set(i, -i - 3000);
        }

        return new DataRow(vector, Vector.of(1, 0, 0, 0, 0, 0));
    }

    public static DataRow getPolar() {
        Vector vector = new Vector(INPUT_DIMENSION);

        vector.set(0, -ThreadLocalRandom.current().nextInt(2000));

        for (int i = 1; i < INPUT_DIMENSION; i++) {
            vector.set(i, -i - 100);
        }

        return new DataRow(vector, Vector.of(0, 1, 0, 0, 0, 0));
    }

    public static DataRow getGrim() {
        Vector vector = new Vector(INPUT_DIMENSION);

        for (int i = 0; i < INPUT_DIMENSION; i++) {
            vector.set(i, -i);
        }

        return new DataRow(vector, Vector.of(0, 0, 1, 0, 0, 0));
    }

    public static DataRow getVulcan() {
        Vector vector = new Vector(INPUT_DIMENSION);

        for (int i = 0; i < INPUT_DIMENSION; i++) {
            vector.set(i, -23767 + i);
        }

        return new DataRow(vector, Vector.of(0, 0, 0, 1, 0, 0));
    }

    public static DataRow getMatrix() {
        Vector vector = new Vector(INPUT_DIMENSION);

        for (int i = 0; i < INPUT_DIMENSION; i++) {
            vector.set(i, i + 100);
        }

        return new DataRow(vector, Vector.of(0, 0, 0, 0, 1, 0));
    }

    public static DataRow getIntave() {
        Vector vector = new Vector(INPUT_DIMENSION);

        for (int i = 0; i < INPUT_DIMENSION; i++) {
            vector.set(i, -((i + 1) % 16));
        }

        return new DataRow(vector, Vector.of(0, 0, 0, 0, 0, 1));
    }

    enum AntiCheat {
        KARHU,
        POLAR,
        GRIM,
        VULCAN,
        MATRIX,
        INTAVE
    }

    static class TestListener extends TrainListener {

        @Override
        public void onEvaluated(DataSet dataSet, int epoch, double loss, long took) {
            System.out.println("Epoch " + epoch + " has loss: " + loss + " in " + took + " ms");
        }
    }
}
