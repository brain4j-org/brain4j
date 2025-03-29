import net.echo.brain4j.Brain4J;
import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.adapters.ModernAdapter;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.loss.Loss;
import net.echo.brain4j.model.impl.Sequential;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.optimizer.impl.AdamW;
import net.echo.math4j.DataSet;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class XorTest {

    public static void main(String[] args) throws Exception {
        new XorTest().testXorModel();
    }

    private void testXorModel() throws Exception {
        Brain4J.setLogging(true);

        var dataSet = getDataSet();
        var model = new Sequential(
                new DenseLayer(2, Activations.LINEAR),
                new DenseLayer(32, Activations.MISH),
                new DenseLayer(1, Activations.SIGMOID)
        );

        model.compile(Loss.BINARY_CROSS_ENTROPY, new AdamW(0.1));

        System.out.println(model.summary());

        var start = System.nanoTime();
        model.fit(dataSet, 1000);
        var took = (System.nanoTime() - start) / 1e6;

        System.out.printf("Trained in %.5f ms%n", took);

        var result = model.evaluate(dataSet);
        double loss = model.loss(dataSet);

        System.out.println(result.confusionMatrix());
        System.out.println("Loss: " + loss);

        assertTrue(loss < 0.01, "Loss is too high! " + loss);

        ModernAdapter.serialize("xor.b4j", model);
    }

    private DataSet<DataRow> getDataSet() {
        DataSet<DataRow> set = new DataSet<>();

        for (int x = 0; x < 2; x++) {
            for (int y = 0; y < 2; y++) {
                Tensor input = TensorFactory.vector(x, y);
                Tensor output = TensorFactory.vector(x ^ y);

                set.add(new DataRow(input, output));
            }
        }

        return set;
    }
}
