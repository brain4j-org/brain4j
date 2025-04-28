package mnist;

import net.echo.brain4j.Brain4J;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.loss.Loss;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.impl.AutoEncoder;
import net.echo.brain4j.training.optimizer.impl.AdamW;
import net.echo.math.activation.Activations;
import net.echo.math.data.ListDataSource;
import net.echo.math.data.Sample;
import net.echo.math.tensor.Tensor;
import net.echo.math.tensor.Tensors;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MNISTAutoEncoder {

    public static void main(String[] args) throws Exception {
        new MNISTAutoEncoder().start();
    }

    public void start() throws Exception {
        Brain4J.setLogging(true);

        ListDataSource trainSource = getDataSource("mnist_train.csv");
        ListDataSource testSource = getDataSource("mnist_test.csv");

        Model model = new AutoEncoder(3,
                new DenseLayer(784, Activations.LINEAR),
                new DenseLayer(16, Activations.RELU),
                new DenseLayer(2, Activations.RELU),
                new DenseLayer(16, Activations.RELU),
                new DenseLayer(784, Activations.LINEAR)
        );

        model.compile(Loss.MEAN_SQUARED_ERROR, new AdamW(0.01));
        System.out.println(model.summary());

//        model.load("mnist-autoencoder.b4j");
//        EvaluationResult result = model.evaluate(testSource);
//        System.out.println(result.confusionMatrix());
        model.fit(trainSource, testSource, 50, 10);
        model.save("mnist-autoencoder");

        Tensor prediction = model.predict(Tensors.random(784));
        System.out.println(Arrays.toString(prediction.shape()));
    }

    public ListDataSource getDataSource(String fileName) throws IOException {
        List<Sample> samples = new ArrayList<>();
        List<String> lines = Files.readAllLines(new File(fileName).toPath());

        for (int i = 1; i < lines.size(); i++) {
            String line = lines.get(i);
            String[] parts = line.split(",");

            float[] inputs = new float[parts.length - 1];

            for (int j = 1; j < inputs.length; j++) {
                inputs[j] = Float.parseFloat(parts[j]);
            }

            Tensor image = Tensors.vector(inputs).div(255);
            samples.add(new Sample(image, image));
        }

        return new ListDataSource(samples, true, 128);
    }
}
