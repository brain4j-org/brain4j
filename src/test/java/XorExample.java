import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.initialization.WeightInit;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.data.DataSet;
import net.echo.brain4j.training.optimizers.impl.Adam;
import net.echo.brain4j.training.optimizers.impl.AdamW;
import net.echo.brain4j.training.techniques.SmartTrainer;
import net.echo.brain4j.training.techniques.TrainListener;
import net.echo.brain4j.training.updater.impl.NormalUpdater;
import net.echo.brain4j.training.updater.impl.StochasticUpdater;
import net.echo.brain4j.utils.Vector;

public class XorExample {

    public static void main(String[] args) {
        XorExample example = new XorExample();
        example.start();
    }

    private void start() {
        Model model = getModel();
        DataSet dataSet = getDataSet();

        dataSet.partition(1);

        SmartTrainer trainer = new SmartTrainer(0.7, 100);

        trainer.addListener(new ExampleListener());
        trainer.startFor(model, dataSet, 10_000);
    }

    private Model getModel() {
        Model model = new Model(
                new DenseLayer(2, Activations.LINEAR),
                new DenseLayer(32, Activations.MISH),
                new DenseLayer(32, Activations.MISH),
                new DenseLayer(1, Activations.SIGMOID)
        );

        return model.compile(WeightInit.HE, LossFunctions.BINARY_CROSS_ENTROPY, new Adam(0.1), new StochasticUpdater());
    }

    private DataSet getDataSet() {
        DataSet set = new DataSet();

        for (int x = 0; x < 2; x++) {
            for (int y = 0; y < 2; y++) {
                int output = x ^ y;

                set.add(Vector.of(x, y), Vector.of(output));
            }
        }

        return set;
    }

    private static class ExampleListener extends TrainListener {

        @Override
        public void onEvaluated(DataSet dataSet, int epoch, double loss, long took) {
            System.out.print("\rEpoch #" + epoch + " Loss " + loss);
        }
    }
}
