import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.adapters.ModernAdapter;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.impl.Sequential;
import net.echo.brain4j.model.initialization.WeightInit;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.evaluation.EvaluationResult;
import net.echo.brain4j.training.optimizers.impl.AdamW;
import net.echo.brain4j.training.updater.impl.StochasticUpdater;
import net.echo.math4j.DataSet;
import net.echo.math4j.math.vector.Vector;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class XorTest {

    @Test
    void testXorModel() throws Exception {
        // Remove the comment if you would like to use the GPU
        // TensorFactory.useGPUIfAvailable();

        // Remove the comment if you want to use a pre-trained model
        // Sequential model = ModernAdapter.deserialize("xor.bin", new Sequential());
        Sequential model = getModel();
        DataSet<DataRow> dataSet = getDataSet();

        System.out.println(model.getStats());

        long start = System.nanoTime();
        model.fit(dataSet, 1000);
        double took = (System.nanoTime() - start) / 1e6;

        System.out.println("Training took: " + took + " ms");

        EvaluationResult result = model.evaluate(dataSet);
        double loss = model.loss(dataSet);

        System.out.println("Loss: " + loss);
        System.out.println(result.confusionMatrix());

        assertTrue(loss < 0.01, "Loss is too high! " + loss);

        ModernAdapter.serialize("xor.bin", model);
    }

    private Sequential getModel() {
        Sequential model = new Sequential(
                new DenseLayer(2, Activations.LINEAR),
                new DenseLayer(32, Activations.MISH),
                new DenseLayer(32, Activations.MISH),
                new DenseLayer(1, Activations.SIGMOID)
        );

        return model.compile(WeightInit.HE, LossFunctions.BINARY_CROSS_ENTROPY, new AdamW(0.1), new StochasticUpdater());
    }

    private DataSet<DataRow> getDataSet() {
        DataSet<DataRow> set = new DataSet<>();

        for (int x = 0; x < 2; x++) {
            for (int y = 0; y < 2; y++) {
                int output = x ^ y;
                set.add(new DataRow(Vector.of(x, y), Vector.of(output)));
            }
        }

        return set;
    }
}
