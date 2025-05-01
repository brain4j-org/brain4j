import org.brain4j.core.Brain4J;
import org.brain4j.core.layer.impl.DenseLayer;
import org.brain4j.core.loss.Loss;
import org.brain4j.core.model.Model;
import org.brain4j.core.model.impl.Sequential;
import org.brain4j.core.training.optimizer.impl.Adam;
import org.brain4j.math.Pair;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.data.Sample;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;

public class BigDataExample {

    public static void main(String[] args) {
        new BigDataExample().start();
    }

    public void start() {
        Brain4J.setLogging(true);
        Model model = new Sequential(
                new DenseLayer(13, Activations.LINEAR),
                new DenseLayer(256, Activations.RELU),
                new DenseLayer(256, Activations.RELU),
                new DenseLayer(256, Activations.RELU),
                new DenseLayer(1, Activations.SIGMOID)
        );

        model.compile(Loss.BINARY_CROSS_ENTROPY, new Adam(0.01));

        System.out.println(model.summary());

        ListDataSource source = getDataSet();
        predict(model, source);

        long start = System.nanoTime();
        model.fit(source, 5);
        long end = System.nanoTime();

        System.out.println("Took for 5 epoches " + (end - start) / 1e6 + " ms");
    }

    public void predict(Model model, ListDataSource source) {
        long start = System.nanoTime();
        ExecutorService executor = Brain4J.getExecutor();

        while (source.hasNext()) {
            Pair<Tensor, Tensor> batch = source.nextBatch();
            executor.submit(() -> model.predict(batch.first()));
        }

        executor.close();
        long end = System.nanoTime();

        System.out.println("---- Prediction ----");
        System.out.println("Input size: " + source.size());
        System.out.println("Time to predict: " + (end - start) / 1e6 + " ms");
    }

    public ListDataSource getDataSet() {
        List<Sample> samples = new ArrayList<>();

        for (int i = 0; i < 327680; i++) {
            Tensor input = Tensors.random(13);
            Tensor output = Tensors.random(1);

            samples.add(new Sample(input, output));
        }

        return new ListDataSource(samples, true, 8192);
    }
}
