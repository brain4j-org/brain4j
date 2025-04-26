import net.echo.brain4j.Brain4J;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.loss.Loss;
import net.echo.brain4j.model.impl.Sequential;
import net.echo.brain4j.training.evaluation.EvaluationResult;
import net.echo.brain4j.training.optimizer.impl.Adam;
import net.echo.math.activation.Activations;
import net.echo.math.data.ListDataSource;
import net.echo.math.data.Sample;
import net.echo.math.tensor.Tensor;
import net.echo.math.tensor.Tensors;

import java.util.ArrayList;
import java.util.List;

public class GuideExample {

    public static void main(String[] args) {
        Brain4J.setLogging(true);
        Sequential model = new Sequential(
                new DenseLayer(2, Activations.LINEAR), // 2 Input neurons
                new DenseLayer(32, Activations.MISH), // 32 Hidden neurons
                new DenseLayer(32, Activations.MISH), // 32 Hidden neurons
                new DenseLayer(1, Activations.SIGMOID) // 1 Output neuron for classification
        );

        model.compile(Loss.BINARY_CROSS_ENTROPY, new Adam(0.1));

        List<Sample> samples = new ArrayList<>();

        for (int x = 0; x < 2; x++) {
            for (int y = 0; y < 2; y++) {
                Tensor input = Tensors.vector(x, y);
                Tensor output = Tensors.vector(x ^ y);

                samples.add(new Sample(input, output));
            }
        }

        ListDataSource dataSet = new ListDataSource(samples, false, 4);
        model.fit(dataSet, 50, 10);

        // You can evaluate the model like this
        EvaluationResult evaluation = model.evaluate(dataSet);
        System.out.println(evaluation.confusionMatrix());
    }
}
