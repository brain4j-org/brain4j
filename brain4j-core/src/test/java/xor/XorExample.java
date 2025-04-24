package xor;

import net.echo.brain4j.Brain4J;
import net.echo.brain4j.clipping.GradientClipper;
import net.echo.brain4j.clipping.impl.L2Clipper;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.layer.impl.LayerNorm;
import net.echo.brain4j.loss.Loss;
import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.impl.Sequential;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.evaluation.EvaluationResult;
import net.echo.brain4j.training.optimizer.impl.AdamW;
import net.echo.brain4j.training.optimizer.impl.GradientDescent;
import net.echo.math.DataSet;
import net.echo.math.Pair;
import net.echo.math.activation.Activations;
import net.echo.math.data.ListDataSource;
import net.echo.math.data.Sample;
import net.echo.math.tensor.Tensor;
import net.echo.math.tensor.Tensors;

import java.util.ArrayList;
import java.util.List;

public class XorExample {

    public static void main(String[] args) {
        new XorExample().start();
    }

    public void start() {
        Brain4J.setLogging(true);
        ListDataSource source = getDataSet();

        Model model = new Sequential(
                new DenseLayer(2, Activations.LINEAR),
                new DenseLayer(32, Activations.MISH),
                new DenseLayer(1, Activations.SIGMOID)
        );

        model.compile(Loss.BINARY_CROSS_ENTROPY, new GradientDescent(0.1));

        double loss = model.loss(source);
        System.out.println("Loss: " + loss);

        long start = System.currentTimeMillis();
        model.fit(source, 100000, 10000);
        long end = System.currentTimeMillis();
        System.out.println("Time: " + (end - start) + " ms");

        loss = model.loss(source);
        System.out.println("Loss: " + loss);
    }

    public ListDataSource getDataSet() {
        List<Sample> samples = new ArrayList<>();

        for (int x = 0; x < 2; x++) {
            for (int y = 0; y < 2; y++) {
                Tensor input = Tensors.vector(x, y);
                Tensor output = Tensors.vector(x ^ y);

                samples.add(new Sample(input, output));
            }
        }

        return new ListDataSource(samples, 1);
    }
}
