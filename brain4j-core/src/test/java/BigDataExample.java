import net.echo.brain4j.Brain4J;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.loss.Loss;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.impl.Sequential;
import net.echo.brain4j.training.optimizer.impl.Adam;
import net.echo.math.activation.Activations;
import net.echo.math.data.AsyncDataSource;
import net.echo.math.data.ListDataSource;
import net.echo.math.data.Sample;
import net.echo.math.tensor.Tensor;
import net.echo.math.tensor.Tensors;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class BigDataExample {

    public static void main(String[] args) {
        new BigDataExample().start();
    }

    public void start() {
        Brain4J.useGPUIfAvailable();
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

        Tensor packet = source.nextBatch().first();

        System.out.println("Shape: " + Arrays.toString(packet.shape()));
        System.out.println("Size: " + packet.elements());

        source.reset();

        long start = System.nanoTime();
        model.fit(source);
        long end = System.nanoTime();

        System.out.println("Took: " + (end - start) / 1e6 + " ms");
    }

    public ListDataSource getDataSet() {
        List<Sample> samples = new ArrayList<>();

        for (int i = 0; i < 327680; i++) {
            Tensor input = Tensors.random(13);
            Tensor output = Tensors.random(1);

            samples.add(new Sample(input, output));
        }

        return new AsyncDataSource(samples, 8);
    }
}
