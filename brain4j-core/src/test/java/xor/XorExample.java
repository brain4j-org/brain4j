package xor;

import net.echo.brain4j.Brain4J;
import net.echo.brain4j.clipping.impl.L2Clipper;
import net.echo.math.activation.Activations;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.layer.impl.LayerNorm;
import net.echo.brain4j.loss.Loss;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.impl.Sequential;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.evaluation.EvaluationResult;
import net.echo.brain4j.training.optimizer.impl.AdamW;
import net.echo.math.DataSet;
import net.echo.math.tensor.Tensor;
import net.echo.math.tensor.Tensors;

public class XorExample {

    public static void main(String[] args) throws Exception {
        new XorExample().start();
    }

    public void start() throws Exception {
        Brain4J.setLogging(true);
        L2Clipper clipper = new L2Clipper(1);

        DataSet<DataRow> dataSet = getDataSet();
        Model model = new Sequential()
                .add(new DenseLayer(2, Activations.LINEAR, clipper))
                .add(new LayerNorm())
                .add(new DenseLayer(32, Activations.MISH, clipper))
                .add(new LayerNorm())
                .add(new DenseLayer(1, Activations.SIGMOID, clipper))
                .compile(Loss.BINARY_CROSS_ENTROPY, new AdamW(0.1));

        System.out.println(model.summary());

        model.fit(dataSet, 100, 20);

        EvaluationResult result = model.evaluate(dataSet);
        System.out.println(result.confusionMatrix());

        model.save("xor.b4j");
    }

    public DataSet<DataRow> getDataSet() {
        DataSet<DataRow> set = new DataSet<>();

        for (int x = 0; x < 2; x++) {
            for (int y = 0; y < 2; y++) {
                Tensor input = Tensors.vector(x, y);
                Tensor output = Tensors.vector(x ^ y);

                set.add(new DataRow(input, output));
            }
        }

        return set;
    }
}
