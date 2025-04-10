import net.echo.brain4j.Brain4J;
import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.adapters.ModernAdapter;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.loss.Loss;
import net.echo.brain4j.model.impl.Sequential;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.evaluation.EvaluationResult;
import net.echo.brain4j.training.optimizer.impl.AdamW;
import net.echo.math4j.DataSet;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class XorApproxExample {

    public static void main(String[] args) throws Exception {
        new XorApproxExample().testXorModel();
    }

    private void testXorModel() throws Exception {
        Brain4J.setLogging(true);

        DataSet<DataRow> dataSet = getDataSet();
        Sequential model = new Sequential(
                new DenseLayer(2, Activations.LINEAR),
                new DenseLayer(32, Activations.MISH),
                new DenseLayer(1, Activations.SIGMOID)
        );

        model.compile(Loss.BINARY_CROSS_ENTROPY, new AdamW(0.1));

        System.out.println(model.summary());

        long start = System.nanoTime();
        model.fit(dataSet, 1000, 100);
        double took = (System.nanoTime() - start) / 1e6;

        System.out.printf("Trained in %.5f ms%n", took);

        EvaluationResult result = model.evaluate(dataSet);
        System.out.println(result.confusionMatrix());

        ModernAdapter.serialize("xor.b4j", model);

        assertTrue(result.loss() < 0.01, "Loss is too high! " + result.loss());
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
