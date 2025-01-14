import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.initialization.WeightInit;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.data.DataSet;
import net.echo.brain4j.training.optimizers.impl.AdamW;
import net.echo.brain4j.training.updater.impl.NormalUpdater;
import net.echo.brain4j.utils.MLUtils;
import net.echo.brain4j.utils.Vector;

public class TransactionsTest {

    private static final int INPUT_DIMENSION = 20;

    public static void main(String[] args) {
        Model model = getModel();

        DataRow karhu = getKarhu();
        DataRow polar = getPolar();
        DataRow grim = getGrim();

        System.out.println(karhu);
        System.out.println(polar);
        System.out.println(grim);

        DataSet training = new DataSet(karhu, polar, grim);
        training.partition(1);

        int i = 0;
        double error = Double.MAX_VALUE;

        do {
            i++;

            model.fit(training);

            if (i % 10000 == 0) {
                error = model.evaluate(training);
                System.out.println("Epoch #" + i + " has error " + error);
            }
        } while (error > 0.01);

        for (DataRow row : training) {
            Vector output = model.predict(row.inputs());

            AntiCheat predicted = MLUtils.findBestMatch(output, AntiCheat.class);
            AntiCheat actual = MLUtils.findBestMatch(row.outputs(), AntiCheat.class);

            System.out.println("-------------------------");
            System.out.println("Inputs: " + row.inputs());
            System.out.println("Output: " + row.outputs());
            System.out.println("Prediction: " + output);
            System.out.println("Actual Anticheat: " + actual);
            System.out.println("Predicted Anticheat: " + predicted);
            System.out.println();
        }
    }

    public static Model getModel() {
        Model model = new Model(
                new DenseLayer(INPUT_DIMENSION, Activations.LINEAR),
                new DenseLayer(8, Activations.SIGMOID),
                new DenseLayer(8, Activations.SIGMOID),
                new DenseLayer(3, Activations.SOFTMAX)
        );

        model.compile(WeightInit.UNIFORM_XAVIER, LossFunctions.CATEGORICAL_CROSS_ENTROPY, new AdamW(0.001), new NormalUpdater());

        return model;
    }

    public static DataRow getKarhu() {
        Vector vector = new Vector(INPUT_DIMENSION);

        for (int i = 0; i < INPUT_DIMENSION; i++) {
            vector.set(i, -i - 3000);
        }

        return new DataRow(vector, Vector.of(1, 0, 0));
    }

    public static DataRow getPolar() {
        Vector vector = new Vector(INPUT_DIMENSION);

        for (int i = 0; i < INPUT_DIMENSION; i++) {
            vector.set(i, -i - 100);
        }

        return new DataRow(vector, Vector.of(0, 1, 0));
    }

    public static DataRow getGrim() {
        Vector vector = new Vector(INPUT_DIMENSION);

        for (int i = 0; i < INPUT_DIMENSION; i++) {
            vector.set(i, -i);
        }

        return new DataRow(vector, Vector.of(0, 0, 1));
    }

    enum AntiCheat {
        KARHU,
        POLAR,
        GRIM
    }
}
