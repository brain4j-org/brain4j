package mnist;

import org.brain4j.core.Brain4J;
import org.brain4j.core.layer.impl.DenseLayer;
import org.brain4j.core.loss.Loss;
import org.brain4j.core.model.Model;
import org.brain4j.core.model.impl.AutoEncoder;
import org.brain4j.core.training.optimizer.impl.AdamW;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.data.Sample;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

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
