import org.brain4j.core.Brain4J;
import org.brain4j.core.layer.impl.DenseLayer;
import org.brain4j.core.loss.Loss;
import org.brain4j.core.model.Model;
import org.brain4j.core.model.impl.Sequential;
import org.brain4j.core.training.evaluation.EvaluationResult;
import org.brain4j.core.training.optimizer.impl.AdamW;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.data.Sample;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

import java.util.ArrayList;
import java.util.List;

public class GuideExample {

    public static void main(String[] args) {
        Brain4J.setLogging(true);
        Model model = new Sequential(
                new DenseLayer(2, Activations.LINEAR), // 2 Input neurons
                new DenseLayer(32, Activations.MISH), // 32 Hidden neurons
                new DenseLayer(32, Activations.MISH), // 32 Hidden neurons
                new DenseLayer(1, Activations.SIGMOID) // 1 Output neuron for classification
        );

        model.compile(Loss.BINARY_CROSS_ENTROPY, new AdamW(0.1));

        List<Sample> samples = new ArrayList<>();

        for (int x = 0; x < 2; x++) {
            for (int y = 0; y < 2; y++) {
                Tensor input = Tensors.vector(x, y);
                Tensor output = Tensors.vector(x ^ y);

                samples.add(new Sample(input, output));
            }
        }

        // Samples, no shuffle, 4 of batch size
        ListDataSource dataSource = new ListDataSource(samples, false, 4);

        // Fit the model for 50 epoches, evaluate every 10
        model.fit(dataSource, 50, 10);

        // You can evaluate the model like this
        EvaluationResult evaluation = model.evaluate(dataSource);
        System.out.println(evaluation.results());
    }
}
